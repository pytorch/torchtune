import torch
from torchao.dtypes.nf4tensor import implements as nf4_tensor_impl, to_nf4


@nf4_tensor_impl([torch.ops.aten.clone.default])
def clone(func, *args, **kwargs):
    return to_nf4(args[0][0].get_original_weight())


@nf4_tensor_impl([torch.ops.aten.copy_.default])
def inplace_copy(func, *args, **kwargs):
    dest_tensor = args[0][0]  # tensor we are inplace copying into
    ref_tensor = to_nf4(
        args[0][1].to(dest_tensor.device)
    )  # TODO check if nf4 tensor takes in device arg
    dest_tensor.block_size = ref_tensor.block_size
    dest_tensor.n_blocks = ref_tensor.n_blocks
    dest_tensor.scaler_block_size = ref_tensor.scaler_block_size
    dest_tensor.quantized_scalers = ref_tensor.quantized_scalers
    dest_tensor.quantization_factor = ref_tensor.quantization_factor
    dest_tensor.scaler_mean = ref_tensor.scaler_mean
    dest_tensor.quantized_data = ref_tensor.quantized_data
    dest_tensor.nf4 = ref_tensor.nf4
