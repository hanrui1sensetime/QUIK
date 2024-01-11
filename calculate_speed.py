import os
import json_lines
from transformers import AutoTokenizer

result_folder = 'predict_w4a4_speed'
tokenizer = '/root/workspace/external_data/tigerbot-13b-base_v9_gpt4_hf'
tokenizer = AutoTokenizer.from_pretrained(tokenizer)

sum = 0
with open(os.path.join(result_folder, 'SmartMR.jsonl'), 'r') as f:
    for line in json_lines.reader(f):
        sum += len(tokenizer.encode(line['predict_answer']))
print(sum)

prompt_template = 'You are PULSE, a large language model of Transformer architecture trained by OpenMedLab. Answer as concisely as possible.\nKnowledge cutoff: 2021-09-01\nCurrent date: 2023-06-28'
data_folder = '/root/workspace/external_data/dayi_data'
answer_start_ids = [tokenizer.convert_tokens_to_ids("<|aim_start|>")]
end_token = '<|im_end|>'
eos_token_ids = [
    tokenizer.convert_tokens_to_ids("<|im_end|>"),
]
suppress_token_ids = [
    tokenizer.eos_token_id,
]

answer_sum = []
question_sum = []
time_dict = {
    'SmartMR': 51 * 60 + 11,
    'webMedQA': 22 * 60 + 30,
    'FollowupChoice': 31 * 60 + 49,
    'HospitalGuide': 2 * 60 + 20,
    'CheckupQA': 20 * 60 + 58,
    'ComprehensiveHealthQA': 28 * 60 + 11,
    'DY_MedicineQA': 25 * 60 + 38,
    'EMR_Structuralization': 27 * 60 + 3,
    'FollowupQuery': 9 * 60 + 46,
    'MedQA-USMLE': 41 * 60 + 37,
    'PromptCBLUE': 17 * 60 + 52,
    'MedicalDecision': 48 * 60 + 46,
    'MedQA_Mainland': 26 * 60 + 35
}
num_question_dict = {
    'SmartMR': 150,
    'webMedQA': 150,
    'FollowupChoice': 150,
    'HospitalGuide': 150,
    'CheckupQA': 150,
    'ComprehensiveHealthQA': 150,
    'DY_MedicineQA': 150,
    'EMR_Structuralization': 150,
    'MedQA_Mainland': 150,
    'FollowupQuery': 150,
    'MedQA-USMLE': 150,
    'PromptCBLUE': 150,
    'MedicalDecision': 150
}
time_sum = []
num_question = []
for file in os.listdir(result_folder):
    sum = 0
    q_sum = 0
    print(f'processing filename: {file}')
    prefix = file.split('.')[0]
    if prefix in time_dict:
        time_sum.append(time_dict[prefix])
    else:
        continue
    if prefix in num_question_dict:
        num_question.append(num_question_dict[prefix])
    else:
        continue
    with open(os.path.join(result_folder, file), 'r') as f:
        for line in json_lines.reader(f):
            question = line['question']
            input_ids = tokenizer(
                f"<|iim_start|>{prompt_template}<|im_end|>").input_ids
            input_ids += tokenizer(f"<|uim_start|>{question}<|im_end|>",
                                   add_special_tokens=False).input_ids
            input_ids += answer_start_ids
            q_sum += len(input_ids)
            sum += len(tokenizer.encode(line['predict_answer']))
    print(f'total answer sum: {sum}')
    print(f'total question sum: {q_sum}')
    question_sum.append(q_sum)
    answer_sum.append(sum)


def cal_time(q_sum, a_sum, t_sum, num_q):
    import numpy as np
    q_sum = np.array(q_sum)
    a_sum = np.array(a_sum)
    t_sum = np.array(t_sum)
    num_q = np.array(num_q)
    print(q_sum.sum(), a_sum.sum(), t_sum.sum())
    q_sum = (q_sum / num_q)**2
    a_sum = a_sum / num_q
    t_sum = t_sum / num_q
    x = np.stack([q_sum, a_sum], axis=1)
    print(x.shape)
    y = t_sum
    index = np.linalg.lstsq(x, y)[0]
    prefill_speed = (1 / index[0])**0.5
    generate_speed = (1 / index[1])
    y = y.reshape(y.shape[0], 1)
    total = np.concatenate([x, y], axis=1)
    print(total.shape)
    relative = np.corrcoef(total, rowvar=False)[:-1, -1]
    print(
        f'prefill_speed: {prefill_speed} token/s, relative coefficient: {relative[0]}, generate_speed: {generate_speed} token/s, relative coefficient: {relative[1]}'
    )


cal_time(question_sum, answer_sum, time_sum, num_question)
