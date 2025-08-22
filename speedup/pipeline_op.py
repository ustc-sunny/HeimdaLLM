# pipeline_op.py
import asyncio
import torch
from typing import Dict, Callable, Awaitable

class PipelineManager:
    """
    事件驱动的两级流水线：
    1) cloud_bp_task   —— 在 cloud 上跑 BP，完成后立即发送梯度到 server；
    2) server_agg_task —— server 聚合 client 梯度，完成后立即与 cloud 梯度加权，
                          然后并行下发给 cloud & client。
    用 asyncio 把 I/O 与计算重叠，后期可换成 Ray / torch.distributed.rpc。
    """
    def __init__(
        self,
        cloud_bp_fn:   Callable[[], Awaitable[Dict[str, torch.Tensor]]],
        server_agg_fn: Callable[[], Awaitable[Dict[str, torch.Tensor]]],
        rectify_fn:    Callable[[Dict, Dict], Dict[str, torch.Tensor]],
        send_fn:       Callable[[Dict[str, torch.Tensor], str], Awaitable[None]]
    ):
        self.cloud_bp_fn   = cloud_bp_fn
        self.server_agg_fn = server_agg_fn
        self.rectify_fn    = rectify_fn
        self.send_fn       = send_fn

    async def run_round(self, alpha: float = 0.5):
        # 1. 并行启动 cloud_bp 与 server 聚合
        cloud_task   = asyncio.create_task(self.cloud_bp_fn())
        server_task  = asyncio.create_task(self.server_agg_fn())

        # 2. 谁先完成都可以继续；这里用 asyncio.wait(fs, return_when=FIRST_COMPLETED)
        done, pending = await asyncio.wait(
            {cloud_task, server_task},
            return_when=asyncio.FIRST_COMPLETED
        )

        # 3. 任一任务完成后立即发送已就绪的梯度
        if cloud_task in done:
            cloud_grad = cloud_task.result()
            await self.send_fn(cloud_grad, "server")
        if server_task in done:
            server_grad = server_task.result()
            await self.send_fn(server_grad, "cloud")

        # 4. 等待剩余任务完成
        if cloud_task not in done:
            cloud_grad = await cloud_task
            await self.send_fn(cloud_grad, "server")
        if server_task not in done:
            server_grad = await server_task
            await self.send_fn(server_grad, "cloud")

        # 5. 修正梯度
        new_grad = self.rectify_fn(server_grad, cloud_grad, alpha)

        # 6. 并行下发
        await asyncio.gather(
            self.send_fn(new_grad, "cloud"),
            self.send_fn(new_grad, "client")
        )