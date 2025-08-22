# zoo_rectifi_op.py
import torch
from typing import Dict

def rectify_gradients(
    server_agg_grad: Dict[str, torch.Tensor],
    cloud_bp_grad:  Dict[str, torch.Tensor],
    alpha: float = 0.5
) -> Dict[str, torch.Tensor]:
    """
    对 server 聚合后的梯度与 cloud bp 梯度做加权平均。
    alpha: server 权重； (1-alpha): cloud 权重
    返回新的全局梯度，用于下发给 client & cloud。
    """
    new_grad = {}
    for key in server_agg_grad.keys():
        new_grad[key] = alpha * server_agg_grad[key] + (1 - alpha) * cloud_bp_grad[key]
    return new_grad