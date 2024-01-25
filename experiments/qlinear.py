import torch
import timeit
import numpy as np
from torch import Tensor
import quik

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def two_compl(x: Tensor, bits: int) -> Tensor:
    return torch.where(x < 0, 2**bits + x, x)


def rev_two_compl(x: Tensor, bits: int) -> Tensor:
    x = x.to(torch.int8)
    return torch.where(x >= 2**(bits - 1), x - 2**bits, x)


def pack_to_i4(X: Tensor):
    X_i8 = two_compl(X.to(dtype=torch.int8), 4).to(torch.uint8)
    X_i4 = X_i8[:, 0::2] | (X_i8[:, 1::2] << 4)
    return X_i4


def pack_to_i8_old(X: Tensor):
    X_i8 = torch.zeros((X.shape[0], X.shape[1] * 2),
                       device=X.device,
                       dtype=torch.uint8,
                       requires_grad=False)
    X_i8[:, 0::2] = (X + 8) & 15
    X_i8[:, 1::2] = ((X + (8 << 4)) >> 4) & 15
    return X_i8


def pack_to_i8(X: Tensor, kvbit=4):
    X_i8 = torch.zeros((X.shape[0], X.shape[1] * 8 // kvbit),
                       device=X.device,
                       dtype=torch.int8,
                       requires_grad=False)
    if kvbit == 4:
        X_i8[:, 0::2] = rev_two_compl(X & 15, 4)
        X_i8[:, 1::2] = rev_two_compl((X >> 4) & 15, 4)
    elif kvbit == 8:
        X_i8 = rev_two_compl(X, 8)
    return X_i8


class SharedQuantizedInput:

    def __init__(self, group_size):
        self.qint_x = None
        self.fp_x = None
        self.qscale_x = None
        self.meta = None
        self.group_size = group_size
        self.cur_group_elem = 0

    def finish(self):
        self.cur_group_elem += 1
        if self.cur_group_elem == self.group_size:
            self.qint_x = None
            self.qscale_x = None
            self.meta = None
            self.fp_x = None
            self.cur_group_elem = 0


class MixedQLinear(torch.nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 shared_input=None,
                 fp_features_num=0,
                 symm=False,
                 bits=4,
                 dtype=torch.float16,
                 kvcache_quant=False,
                 kvcache_bit=4):
        super().__init__()
        self.fp_features_num = fp_features_num
        self.int_features_num = in_features - fp_features_num
        self.in_features = in_features
        self.out_features = out_features
        self.symmetric = symm
        self.bits = bits
        self.shared_input = shared_input
        self.dtype = dtype
        self.kvcache_quant = kvcache_quant
        # self.kvcache_shared_input = None
        self.kvcache_meta = None
        self.kvcache_bit = kvcache_bit
        self.register_buffer(
            'weights_scales',
            torch.zeros((self.out_features, 1),
                        dtype=self.dtype,
                        requires_grad=False))
        # Split for quantized weights
        if self.bits == 4:
            self.register_buffer(
                'int_weight',
                torch.randint(
                    1,
                    7,
                    (self.out_features, self.int_features_num // 2),
                    # SubByte weight
                    dtype=torch.uint8,
                    requires_grad=False))
        else:
            self.register_buffer(
                'int_weight',
                torch.randint(-128,
                              127, (self.out_features, self.int_features_num),
                              dtype=torch.int8,
                              requires_grad=False))

        self.register_buffer(
            'bias',
            torch.zeros((self.out_features), dtype=dtype, requires_grad=False))

        self.register_buffer(
            'int_indices',
            torch.zeros((self.int_features_num),
                        dtype=torch.long,
                        requires_grad=False))
        self.register_buffer(
            'fp_indices',
            torch.zeros((self.fp_features_num),
                        dtype=torch.long,
                        requires_grad=False))

        if self.fp_features_num > 0:
            # Split for full precision weights
            self.register_buffer(
                'fp_weight',
                torch.randint(-8,
                              7, (self.out_features, self.fp_features_num),
                              dtype=dtype,
                              requires_grad=False))
        if not self.symmetric:
            self.register_buffer('reduced_w',
                                 torch.zeros((1, self.out_features),
                                             dtype=dtype,
                                             requires_grad=False))  # Reduced

    def forward(self, x):
        if self.int_features_num <= 0:
            return torch.nn.functional.linear(x, self.fp_weight, self.bias)
        if torch.cuda.current_device() != x.device.index:  # fix bug
            torch.cuda.set_device(x.device)

        shared_input = self.shared_input
        if shared_input is None:
            shared_input = SharedQuantizedInput(1)
        '''
        if len(x.shape) == 3:
            x = x[0]
        '''

        batch_size = x.shape[0]
        x = x.reshape(1, -1, x.shape[-1])[0]

        if shared_input.qint_x is None:

            # Quantize the int part of the input
            if self.symmetric:
                if self.fp_features_num > 0:
                    int_x = x[:, self.int_indices]
                    shared_input.fp_x = x[:, self.fp_indices]
                else:
                    int_x = x
                reshaped_x = int_x.reshape((-1, int_x.shape[-1]))
                shared_input.qscale_x = (
                    torch.max(torch.abs(reshaped_x), dim=1)[0].unsqueeze(1) /
                    (1 << (self.bits - 1) - 1)).to(torch.float16)
                shared_input.qint_x = quik.symmetric.quantize(
                    int_x, shared_input.qscale_x)
            # elif self.fp_features_num > 0:
            else:
                # if self.fp_features_num > 0:
                #     int_x = x[:, self.int_indices]
                #     shared_input.fp_x = x[:, self.fp_indices]
                # else:
                #     int_x = x
                # shared_input.meta = quik.asymmetric.find_meta(int_x, self.bits)
                # shared_input.qint_x = quik.asymmetric.quantizeOld(int_x, shared_input.meta, self.bits)
                # x = x.reshape(1, -1, 5120)  ## for multi-batch
                meta = torch.zeros((0),
                                   dtype=torch.float16,
                                   device=x.device,
                                   requires_grad=False)
                shared_input.qint_x, shared_input.meta, shared_input.fp_x = quik.asymmetric.quantize(
                    x, self.int_indices, self.fp_indices, meta, self.bits)
            # else:
            #     shared_input.meta = quik.asymmetric.find_meta(x, self.bits)
            #     shared_input.qint_x = quik.asymmetric.quantizeOld(x, shared_input.meta, self.bits)

        # Compute matmul for full precision part
        if self.fp_features_num > 0:
            fp_result = torch.nn.functional.linear(shared_input.fp_x,
                                                   self.fp_weight, self.bias)
        elif self.bias is not None:
            fp_result = self.bias.repeat(shared_input.qint_x.shape[0], 1)
        else:
            if not hasattr(self, "zeros_add"):
                self.register_buffer(
                    "zeros_add",
                    torch.zeros((shared_input.qint_x.shape[0],
                                 self.int_weight.shape[0]),
                                dtype=self.dtype,
                                requires_grad=False,
                                device=self.int_weight.device))
            fp_result = self.zeros_add

        # Compute matmul for int part
        if self.bits == 4:
            int_result = quik.matmul.int4Matmul(shared_input.qint_x,
                                                self.int_weight)
        else:
            int_result = quik.matmul.int8Matmul(shared_input.qint_x,
                                                self.int_weight)
        # Dequantize result and add to full precision part
        if self.symmetric:
            output = quik.symmetric.dequantize(int_result,
                                               shared_input.qscale_x,
                                               self.weights_scales, fp_result)
        else:
            output = quik.asymmetric.dequantize(int_result, shared_input.meta,
                                                self.weights_scales,
                                                self.reduced_w, fp_result,
                                                self.bits)
        shared_input.finish()
        #output = output.reshape((1, *output.shape))
        output = output.reshape((batch_size, -1, output.shape[-1]))
        return output

    def quantize_kv(self, x, position_ids):
        '''
        if self.kvcache_meta is None:
            meta = torch.zeros((x.shape[-2] * 2), dtype=torch.float16, requires_grad=False).to(x.device)
        '''

        # shared_input = self.kvcache_shared_input
        # if shared_input is None:
        # shared_input = SharedQuantizedInput(1)
        head_dim = x.shape[-1]
        bsz, num_heads, seq_len, head_dim = x.shape
        # x = x.permute(1, 2, 0, 3)
        # x = x.reshape(-1, bsz * head_dim)
        x = x.transpose(1, 0)
        x = x.reshape(-1, head_dim)

        # cache_int_indices = torch.arange((bsz * head_dim), dtype=torch.long, requires_grad=False).to(x.device)
        cache_int_indices = torch.arange((head_dim),
                                         dtype=torch.long,
                                         requires_grad=False).to(x.device)
        cache_fp_indices = torch.zeros((0),
                                       dtype=torch.long,
                                       requires_grad=False).to(x.device)
        '''
        if self.kvcache_meta is None:
            meta = torch.zeros((seq_len * num_heads * 2), dtype=torch.float16,
                               requires_grad=False, device=x.device)
            qint_x, meta, _ = quik.asymmetric.quantize(x,
                                                       cache_int_indices,
                                                       cache_fp_indices,
                                                       meta,
                                                       self.kvcache_bit,
                                                       quant_static=False)
        else:
            qint_x, meta, _ = quik.asymmetric.quantize(x,
                                                       cache_int_indices,
                                                       cache_fp_indices,
                                                       self.kvcache_meta,
                                                       self.kvcache_bit,
                                                       quant_static=True)
            assert torch.allclose(meta, self.kvcache_meta)
        '''
        '''
        if start_pos == 0:
            meta = self.kvcache_meta.reshape(num_heads, -1, 2)[:, :seq_len, :].contiguous().flatten().to(x.device)
        else:
            meta = self.kvcache_meta.reshape(num_heads, -1, 2)[:, start_pos:start_pos+seq_len, :].contiguous().flatten().to(x.device)
        '''
        meta = self.kvcache_meta.to(x.device).reshape(num_heads, -1,
                                                      2)[:, position_ids, :]
        meta = meta.contiguous().flatten()

        qint_x, meta, _ = quik.asymmetric.quantize(x,
                                                   cache_int_indices,
                                                   cache_fp_indices,
                                                   meta,
                                                   self.kvcache_bit,
                                                   quant_static=True)
        del cache_int_indices
        del cache_fp_indices
        '''
        if shared_input.qint_x is not None:
            shared_input.qint_x = torch.cat([shared_input.qint_x, qint_x], dim=0)
        else:
            shared_input.qint_x = qint_x
        if shared_input.meta is not None:
            shared_input.meta = torch.cat([shared_input.meta, meta])
        else:
            shared_input.meta = meta
        '''
        '''
        if self.kvcache_meta is not None:
            self.kvcache_meta = torch.cat([self.kvcache_meta, meta])
        else:
            self.kvcache_meta = meta
        del meta
        '''
        return qint_x, meta

    def dequantize_kv(self, x, position_ids):
        '''
        bsz = x.shape[0]
        x = x.permute(1, 2, 0, 3)
        x = x.reshape(-1, bsz * x.shape[-1])
        '''
        seq_len = x.shape[2]
        if position_ids.shape[1] < seq_len:
            m = position_ids.max()
            pos = torch.arange(m + 1, device=x.device) - m + position_ids
            position_ids = torch.where(pos < 0, 1, pos)
            del pos
        meta = self.kvcache_meta.to(x.device).reshape(
            x.shape[0], -1, 2)[:, position_ids, :].contiguous().flatten()
        x = x.reshape(-1, x.shape[-1])
        x = pack_to_i8(x, self.kvcache_bit).to(torch.int)
        weights_scales = torch.ones((x.shape[-1], 1),
                                    dtype=torch.float16,
                                    requires_grad=False).to(x.device)
        fp_result = torch.zeros((x.shape[0], x.shape[1]),
                                dtype=torch.float16,
                                requires_grad=False).to(x.device)
        reduced_w = torch.ones((1, x.shape[-1]),
                               dtype=torch.float16,
                               requires_grad=False).to(x.device)
        output = quik.asymmetric.dequantize(x, meta, weights_scales, reduced_w,
                                            fp_result, self.kvcache_bit)
        del weights_scales
        del fp_result
        del reduced_w
        return output

    @staticmethod
    def from_float(module: torch.nn.Linear,
                   weight_matrix,
                   weights_scales,
                   shared_input=None,
                   fp_indices=None,
                   symm=False,
                   bits=4,
                   kvcache_quant=False,
                   kvcache_bit=4):
        '''
        Generate a new MixedQLinear module from a FP16 Linear module.
        The weight matrix should have the same shape as the weight matrix of the FP16 Linear module and rounded using torch.round()
        routine. We will convert it to subByte representation and save it in the int_weight buffer.
        The FP16 weights will be saved in the fp_weight buffer.
        '''
        assert weights_scales.shape == (
            module.out_features,
            1), 'weights_scales should have shape (out_features, 1)'
        assert weight_matrix.shape == (module.out_features, module.in_features)
        assert (
            symm and bits == 4
        ) or not symm, "Symmetric quantization with 8 bits is not supported"
        int_indices = torch.arange(module.in_features)
        if fp_indices is None or len(fp_indices) == 0:
            fp_indices = torch.tensor([], dtype=int_indices.dtype)
        else:
            int_indices = int_indices[~torch.isin(int_indices, fp_indices)]

        assert torch.numel(int_indices) + torch.numel(
            fp_indices
        ) == module.in_features, 'There are some duplication in the fp_indices!'

        int_module = MixedQLinear(module.in_features,
                                  module.out_features,
                                  shared_input,
                                  fp_features_num=torch.numel(fp_indices),
                                  symm=symm,
                                  bits=bits,
                                  dtype=weight_matrix.dtype,
                                  kvcache_quant=kvcache_quant,
                                  kvcache_bit=kvcache_bit)

        weight_matrix = weight_matrix.cuda()
        int_module.weights_scales.copy_(weights_scales.to(weight_matrix.dtype))
        int_rounded_weight = (weight_matrix[:, int_indices] /
                              weights_scales.to(weight_matrix.device)).round()
        if bits == 4:
            int_module.int_weight.copy_(
                pack_to_i4(int_rounded_weight.to(torch.int8)).cpu())
        else:
            int_module.int_weight.copy_(
                int_rounded_weight.to(torch.int8).cpu())
        if not symm:
            # reduced_w = torch.sum(int_rounded_weight.float(), dim=1, keepdim=True).half()
            reduced_w = torch.sum(weight_matrix[:, int_indices].float(),
                                  dim=1,
                                  keepdim=True).to(weight_matrix.dtype)
            int_module.reduced_w.copy_(reduced_w.t().cpu())
            # if torch.isinf(reduced_w).sum() > 0 or torch.isnan(reduced_w).sum() > 0:
            #     print("Bad reduced w")

        if module.bias is not None:
            int_module.bias.copy_(module.bias)
        else:
            int_module.bias = None
        int_module.int_indices.copy_(int_indices)
        int_module.fp_indices.copy_(fp_indices)

        if int_module.fp_features_num > 0:
            int_module.fp_weight.copy_(weight_matrix[:, fp_indices].to(
                weight_matrix.dtype))
        return int_module


class Linear8bit(torch.nn.Module):

    def __init__(self, in_features, out_features, bias=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            'int_weight',
            torch.randint(
                -128,
                127,
                (self.out_features, self.in_features),
                # SubByte weight
                dtype=torch.int8,
                requires_grad=False))
        self.register_buffer(
            'bias',
            torch.zeros((self.out_features),
                        dtype=torch.float16,
                        requires_grad=False))
        self.maxq = torch.tensor(255)
        if bias is not None:
            self.bias.copy_(bias)
        else:
            self.bias = None

    def forward(self, x):
        # Quantize the int part of the input
        if len(x.shape) == 3:
            x = x[0]

        x_int8 = x.to(torch.int8)
        out = quik.matmul.int8Matmul(x_int8, self.int_weight).to(torch.float16)
        if self.bias is not None:
            return out.add(self.bias)
        else:
            return out


class Linear4bit(torch.nn.Module):

    def __init__(self, in_features, out_features, bias=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            'int_weight',
            torch.randint(
                0,
                15,
                (self.out_features, self.in_features // 2),
                # SubByte weight
                dtype=torch.uint8,
                requires_grad=False))
        self.register_buffer(
            'bias',
            torch.zeros((self.out_features),
                        dtype=torch.float16,
                        requires_grad=False))
        if bias is not None:
            self.bias.copy_(bias)
        else:
            self.bias = None

    def forward(self, x):
        # Quantize the int part of the input
        if len(x.shape) == 3:
            x = x[0]

        x_int4 = x[:, :x.size(1) // 2].to(torch.uint8)
        out = quik.matmul.int4Matmul(x_int4, self.int_weight).to(torch.float16)
        if self.bias is not None:
            return out.add(self.bias)
        else:
            return out
