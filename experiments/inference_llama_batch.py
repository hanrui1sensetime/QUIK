import torch
import modelutils
import quant_sim
import qlinear
from llama import llama_multigpu
import os
from transformers import AutoTokenizer, GenerationConfig
import glob
from tqdm import tqdm
import json
from datasets import load_dataset

batch_size = 4
load_qmodel_path = "/root/workspace/external_data/tigerbot-13b-base_v9_gpt4_hf/13bv9.1_quik_w4a4.bin"
model_path = "/root/workspace/external_data/tigerbot-13b-base_v9_gpt4_hf"
# data_path = "/root/workspace/external_data/dayi_data"
data_path = "dayi_data"
predict_output_path = "/home/QUIK/predict_w4a4_speed_batch"
seq_len = 16384
w_bits = 4
a_bits = 4
int8_down_proj = True
fp_features_num = 256
fp_features_frac = None
DEV = torch.device('cuda:0') if torch.cuda.is_available() else torch.device(
    'cpu')

if fp_features_num > 0 or fp_features_frac is not None:
    relative_path = "experiments/act_scales/{}.pt".format(
        model_path.split('/')[-1])
    assert os.path.exists(relative_path)
    act_scales = torch.load(relative_path)
else:
    act_scales = None

model = modelutils.get_llama(model_path, seq_len, '')
save_dict = torch.load(load_qmodel_path)
model.load_state_dict(save_dict["model"])


def llama_replace_with_kernels(model, args):
    model.lm_head.module.weight.requires_grad = False
    model.model.embed_tokens.weight.requires_grad = False
    model.model.norm.weight.requires_grad = False
    layers = model.model.layers
    shared_inputs = {}

    assert not args.w_asym, 'Benchmarking only supports symmetric weight quantization!'
    print("Replace with INT4 kernels.")
    for i in range(len(layers)):
        opt_block = layers[i]
        opt_block.post_attention_layernorm.weight.requires_grad = False
        opt_block.input_layernorm.weight.requires_grad = False
        sequential = [[
            'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'
        ], ['self_attn.o_proj'], ['mlp.up_proj', 'mlp.gate_proj'],
                      ['mlp.down_proj']]
        full = modelutils.find_layers(opt_block)
        for j, layer_group in enumerate(sequential):
            subset = {n: full[n] for n in layer_group}
            shared_inputs[f"{i}.{j}"] = qlinear.SharedQuantizedInput(
                len(layer_group))
            for name in subset:
                layer = subset[name]
                if 'lm_head' in name or 'rotary_emb' in name:
                    continue
                is_quantized = False
                bits = 16
                fp_features = 0
                if isinstance(layer, quant_sim.ActQuantWrapper):
                    if layer.quantizer.configured:
                        is_quantized = True
                        bits = layer.quantizer.bits
                        fp_features = layer.fp_features_num
                    layer = layer.module
                layer_weight = layer.weight.data

                layer_scale = save_dict['model.layers.{}.{}.scale'.format(
                    i, name)]
                if fp_features == 0:
                    fp_feature_idx = None
                else:
                    layer_act_scales = act_scales['model.layers.{}.{}'.format(
                        i, name)]
                    fp_feature_idx = torch.sort(
                        layer_act_scales)[1][-fp_features:]

                if is_quantized:
                    kvcache_quant = False
                    if 'k_proj' in name or 'v_proj' in name:
                        kvcache_quant = True
                        kvcache_bit = 4
                    int_mod = qlinear.MixedQLinear.from_float(
                        layer,
                        layer_weight,
                        layer_scale,
                        shared_inputs[f"{i}.{j}"],
                        fp_feature_idx,
                        bits=bits,
                        kvcache_quant=kvcache_quant,
                        kvcache_bit=kvcache_bit)
                else:
                    int_mod = layer
                modelutils.replace_single_mod_opt(opt_block, name, int_mod)


quant_sim.add_actquant(model)
layers = modelutils.find_layers(model)

for name in layers:
    bits = a_bits
    if 'lm_head' in name or "rotary_emb" in name:
        print(f'Skipping {name}\n')
        continue

    if 'down_proj' in name:
        if int8_down_proj:
            bits = 8

    if fp_features_num > 0 or fp_features_frac is not None:
        if fp_features_frac is not None:
            fp_features_num = max(int(layers[name].module * fp_features_frac),
                                  fp_features_num)

        if "qkv" in name:
            act_name = name.replace("qkv", "q")
        else:
            act_name = name
        layers[name].fp_features_configure(act_scales[act_name],
                                           fp_features_num)
    layers[name].quantizer.configure(bits=bits)
if w_bits < 16 and a_bits < 16:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--w_asym',
        type=str,
        help='w_asym',
        default=False,
    )
    args = parser.parse_args()
    llama_replace_with_kernels(model, args)

gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
if len(gpus) > 1:
    llama_multigpu(model, gpus)
else:
    model = model.to(DEV)

tokenizer = AutoTokenizer.from_pretrained(model_path,
                                          trust_remote_code=True,
                                          padding_side='left')

# Convert prompt to tokens
prompt_template = 'You are PULSE, a large language model of Transformer architecture trained by OpenMedLab. Answer as concisely as possible.\nKnowledge cutoff: 2021-09-01\nCurrent date: 2023-06-28'

answer_start_ids = [tokenizer.convert_tokens_to_ids("<|aim_start|>")]
end_token = '<|im_end|>'
eos_token_ids = [
    tokenizer.convert_tokens_to_ids("<|im_end|>"),
]
suppress_token_ids = [
    tokenizer.eos_token_id,
]
max_retry = 1
for test_file_path in sorted(
        glob.glob(os.path.join(data_path, "**/*.jsonl"), recursive=True)):
    predict_file_path = test_file_path.replace(data_path, predict_output_path)
    print(f"run eval on {test_file_path}")
    print(f"save eval on {predict_file_path}")

    if os.path.exists(predict_file_path):
        print(f"{predict_file_path} is finish, continue")
        continue

    test_dataset = load_dataset(
        "json",
        data_files=test_file_path,
        split="train",
    )
    predict_output = []
    batch_list = []
    for data in tqdm(test_dataset):
        retry = 0
        question = data['question']
        query = f"<|iim_start|>{prompt_template}<|im_end|>" + f"<|uim_start|>{question}<|im_end|>" + "<|aim_start|>"
        batch_list.append(query)
        if len(batch_list) < batch_size:
            if len(predict_output) + len(batch_list) < test_dataset.shape[0]:
                continue
            else:
                batch_list += [
                    f"<|iim_start|>{prompt_template}<|im_end|>" +
                    "<|uim_start|><|im_end|>" + "<|aim_start|>"
                ] * (batch_size - test_dataset.shape[0] % batch_size)

        input_ids = torch.tensor(
            tokenizer(batch_list, add_special_tokens=False,
                      padding=True).input_ids).cuda()
        '''
        input_ids = tokenizer(f"<|iim_start|>{prompt_template}<|im_end|>").input_ids
        input_ids += tokenizer(f"<|uim_start|>{question}<|im_end|>", add_special_tokens=False).input_ids
        input_ids += answer_start_ids
        '''

        # input_ids = data['input_ids']
        start_pos = input_ids.shape[-1]
        tokens = input_ids
        generation_config = GenerationConfig(
            max_length=16384,
            max_new_tokens=2048,
            num_beams=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_ids,
            suppress_tokens=suppress_token_ids)
        try:
            for layer in model.model.model.layers:
                layer.self_attn.start_pos = 0
        except Exception:
            for layer in model.model.layers:
                layer.self_attn.start_pos = 0

        while retry < max_retry:
            try:
                model = model.eval()
                generation_output = model.generate(
                    tokens,
                    generation_config,
                )
                break
            except Exception as e:
                print(e)
                retry += 1
                print('retry')

        predict_output += generation_output[:, start_pos:]
        batch_list = []
    predict_output = predict_output[:test_dataset.shape[0]]
    os.makedirs(os.path.dirname(predict_file_path), exist_ok=True)

    with open(predict_file_path, "w", encoding="utf8") as f:
        for test_dataset_item, predict_output_item in zip(
                test_dataset, predict_output):
            f.write(
                json.dumps(
                    {
                        "type":
                        test_dataset_item["type"],
                        "question":
                        test_dataset_item["question"],
                        "reference_answer":
                        test_dataset_item["reference_answer"],
                        "predict_answer":
                        tokenizer.decode(predict_output_item).strip().split(
                            end_token)[0],
                    },
                    ensure_ascii=False) + "\n")
