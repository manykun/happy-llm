# -*- coding: utf-8 -*-
"""DDP 训练通用工具。

本文件只放和“分布式训练”相关的公共逻辑，预训练与 SFT 脚本可以共用。
初学者可以重点理解三个概念：
1. rank：当前进程编号；rank 0 通常负责打印日志和保存模型。
2. local_rank：当前进程在本机使用的 GPU 编号；两卡训练时通常是 0 或 1。
3. world_size：总进程数；单机两卡 DDP 时 world_size=2。
"""

import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


@dataclass
class DistributedContext:
    """保存当前训练进程的分布式信息。"""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    is_distributed: bool

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def setup_distributed(args) -> DistributedContext:
    """初始化 DDP 环境，并为当前进程绑定 GPU。

    推荐启动方式：
        torchrun --standalone --nproc_per_node=2 ddp_pretrain.py

    torchrun 会自动写入 RANK、LOCAL_RANK、WORLD_SIZE 等环境变量。
    如果没有这些变量，本函数会退化为普通单进程训练，方便本地调试。
    """

    # 如果用户传入 --gpus 0,1，则只让当前进程看到这几张卡。
    # 注意：最稳妥的方式仍然是在命令前写 CUDA_VISIBLE_DEVICES=0,1。
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1

    if is_distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP 多卡训练需要 CUDA，请检查 GPU 环境。")

        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=args.ddp_backend)
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        local_rank = 0
        if args.device == "auto":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)

    args.device = str(device)
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        is_distributed=is_distributed,
    )


def cleanup_distributed(ddp_ctx: DistributedContext) -> None:
    """训练结束后关闭进程组。"""

    if ddp_ctx.is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def rank0_print(ddp_ctx: DistributedContext, content: str) -> None:
    """只在 rank 0 打印，避免两张卡重复输出同一条日志。"""

    if ddp_ctx.is_main_process:
        print(content, flush=True)


def seed_everything(seed: int, rank: int = 0) -> None:
    """设置随机种子。

    每个 rank 加上自己的编号，避免所有 DataLoader worker 产生完全相同的随机序列。
    """

    real_seed = seed + rank
    random.seed(real_seed)
    np.random.seed(real_seed)
    torch.manual_seed(real_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(real_seed)


def create_train_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    ddp_ctx: DistributedContext,
    drop_last: bool = False,
) -> tuple[DataLoader, Optional[DistributedSampler]]:
    """创建训练 DataLoader。

    DDP 必须使用 DistributedSampler，让不同进程读取不同的数据切片。
    如果继续直接 shuffle=True，每张卡会看到几乎相同的数据，训练就不是“真正的 DDP”。
    """

    sampler = None
    if ddp_ctx.is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=ddp_ctx.world_size,
            rank=ddp_ctx.rank,
            shuffle=True,
            drop_last=drop_last,
        )

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        pin_memory=(ddp_ctx.device.type == "cuda"),
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return train_loader, sampler


def wrap_model_for_ddp(model: torch.nn.Module, ddp_ctx: DistributedContext) -> torch.nn.Module:
    """把模型移动到当前 GPU，并在多进程训练时包装成 DDP。"""

    model = model.to(ddp_ctx.device)
    if ddp_ctx.is_distributed:
        model = DDP(
            model,
            device_ids=[ddp_ctx.local_rank],
            output_device=ddp_ctx.local_rank,
            find_unused_parameters=False,
        )
    return model


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """从 DDP/DataParallel 包装中取出原始模型，用于保存权重。"""

    return model.module if hasattr(model, "module") else model


def save_checkpoint(model: torch.nn.Module, path: str, ddp_ctx: DistributedContext) -> None:
    """只在 rank 0 保存模型，避免多个进程同时写同一个文件。"""

    if not ddp_ctx.is_main_process:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(unwrap_model(model).state_dict(), path)


def reduce_mean(tensor: torch.Tensor, ddp_ctx: DistributedContext) -> torch.Tensor:
    """把每个 rank 的指标求平均，常用于打印全局平均 loss。"""

    if not ddp_ctx.is_distributed:
        return tensor.detach()

    reduced = tensor.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= ddp_ctx.world_size
    return reduced
