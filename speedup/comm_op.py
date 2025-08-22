# comm_op.py
import torch
from typing import Dict, Tuple
import zlib, pickle, math

def compress_state_dict(
    tensor_dict: Dict[str, torch.Tensor], budget_bytes: int = 64 * 1024
) -> bytes:
    """
    简单示例：先 pickle → zlib，后期可换成 Top-K / Quant-8。
    这里先用固定字节预算触发异常，方便你后续替换算法。
    """
    data = pickle.dumps({k: v.cpu().float() for k, v in tensor_dict.items()})
    data = zlib.compress(data, level=6)
    if len(data) > budget_bytes:
        raise RuntimeError(f"compressed {len(data)} > budget {budget_bytes}")
    return data

def decompress_state_dict(blob: bytes) -> Dict[str, torch.Tensor]:
    data = zlib.decompress(blob)
    return pickle.loads(data)