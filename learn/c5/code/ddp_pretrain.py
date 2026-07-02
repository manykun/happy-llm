# -*- coding: utf-8 -*-
"""Tiny-LLM 预训练脚本，支持 PyTorch DDP。

两卡 4090 推荐启动方式：
    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 ddp_pretrain.py

DDP 的核心思想：每张 GPU 启动一个 Python 进程，每个进程只负责自己那张卡；
梯度在 backward 时自动 all-reduce，同步后每张卡上的模型参数保持一致。
"""

import argparse
import importlib
import math
import os
import time
import warnings
from contextlib import nullcontext

import torch
from torch import optim
from transformers import AutoTokenizer

from dataset import PretrainDataset
from k_model import ModelConfig, Transformer
from train_ddp_utils import (
    cleanup_distributed,
    create_train_dataloader,
    rank0_print,
    reduce_mean,
    save_checkpoint,
    seed_everything,
    setup_distributed,
    wrap_model_for_ddp,
)

swanlab = None


warnings.filterwarnings("ignore")


def logger(content: str) -> None:
    """只在 rank 0 打印日志，避免多卡时重复刷屏。"""

    rank0_print(ddp_ctx, content)


def get_lr(step: int, total_steps: int) -> float:
    """余弦退火学习率。

    训练初期可选 warmup：学习率从 0 线性增加到 learning_rate；
    后续按余弦曲线逐渐下降到 learning_rate / 10。
    """

    warmup_iters = args.warmup_iters
    min_lr = args.learning_rate / 10

    if warmup_iters > 0 and step < warmup_iters:
        return args.learning_rate * step / warmup_iters

    if total_steps <= warmup_iters:
        return args.learning_rate

    if step > total_steps:
        return min_lr

    decay_ratio = (step - warmup_iters) / (total_steps - warmup_iters)
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (args.learning_rate - min_lr)


def train_epoch(epoch: int) -> None:
    """训练一个 epoch。

    DDP 下每个进程只看到自己那份数据；反向传播时 DDP 会自动同步梯度。
    梯度累积时，非更新步使用 model.no_sync()，减少不必要的通信。
    """

    if train_sampler is not None:
        # 让 DistributedSampler 每个 epoch 使用不同的 shuffle 顺序。
        train_sampler.set_epoch(epoch)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()

    for step, (x, y, loss_mask) in enumerate(train_loader):
        x = x.to(args.device, non_blocking=True)
        y = y.to(args.device, non_blocking=True)
        loss_mask = loss_mask.to(args.device, non_blocking=True)

        global_step = epoch * iter_per_epoch + step
        lr = get_lr(global_step, args.epochs * iter_per_epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        accum_start = (step // args.accumulation_steps) * args.accumulation_steps
        current_accum_steps = min(args.accumulation_steps, iter_per_epoch - accum_start)
        is_update_step = (step + 1) % args.accumulation_steps == 0 or (step + 1) == iter_per_epoch
        sync_context = model.no_sync() if ddp_ctx.is_distributed and not is_update_step else nullcontext()

        with sync_context:
            with autocast_ctx:
                out = model(x, y)
                token_loss = out.last_loss
                loss_mask = loss_mask.reshape(-1).to(dtype=token_loss.dtype)
                valid_tokens = loss_mask.sum().clamp_min(1.0)
                loss = (token_loss * loss_mask).sum() / valid_tokens
                loss = loss / current_accum_steps

            scaler.scale(loss).backward()

        if is_update_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            eta_minutes = spend_time / (step + 1) * (iter_per_epoch - step - 1) / 60
            loss_for_log = reduce_mean(loss.detach() * current_accum_steps, ddp_ctx)
            logger(
                "Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} eta:{:.1f}min".format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss_for_log.item(),
                    optimizer.param_groups[-1]["lr"],
                    eta_minutes,
                )
            )

            if args.use_swanlab and ddp_ctx.is_main_process:
                swanlab.log({"loss": loss_for_log.item(), "lr": optimizer.param_groups[-1]["lr"]})

        if (step + 1) % args.save_interval == 0:
            ckp = os.path.join(
                args.save_dir,
                f"pretrain_{lm_config.dim}_{lm_config.n_layers}_{lm_config.vocab_size}.pth",
            )
            save_checkpoint(model, ckp, ddp_ctx)

        if (step + 1) % 20000 == 0:
            ckp = os.path.join(
                args.save_dir,
                f"pretrain_{lm_config.dim}_{lm_config.n_layers}_{lm_config.vocab_size}_step{step + 1}.pth",
            )
            save_checkpoint(model, ckp, ddp_ctx)


def init_model() -> tuple[torch.nn.Module, AutoTokenizer]:
    """初始化 tokenizer 和模型，并在需要时包装为 DDP。"""

    def count_parameters(module: torch.nn.Module) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token_id is not None:
        lm_config.pad_token_id = tokenizer.pad_token_id

    model = Transformer(lm_config)
    model = wrap_model_for_ddp(model, ddp_ctx)

    logger(
        f"DDP状态：distributed={ddp_ctx.is_distributed}, "
        f"rank={ddp_ctx.rank}, local_rank={ddp_ctx.local_rank}, world_size={ddp_ctx.world_size}"
    )
    logger(f"LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万")
    return model, tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny-LLM DDP Pretraining")

    parser.add_argument("--out_dir", type=str, default="base_model_215M", help="模型输出目录")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="每张 GPU 上的 batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--device", type=str, default="auto", help="单进程调试设备，例如 auto/cuda:0/cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"], help="混合精度类型")

    parser.add_argument("--use_swanlab", action="store_true", help="是否使用 SwanLab 记录实验")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader 工作进程数")
    parser.add_argument("--data_path", type=str, default="./seq_monkey_datawhale.jsonl", help="预训练数据路径")
    parser.add_argument("--tokenizer_path", type=str, default="./tokenizer_k/", help="tokenizer 路径")

    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--warmup_iters", type=int, default=0, help="学习率 warmup 步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="checkpoint 保存间隔")

    parser.add_argument("--gpus", type=str, default="", help="可见 GPU，例如 0,1；推荐用 CUDA_VISIBLE_DEVICES 设置")
    parser.add_argument("--ddp_backend", type=str, default="nccl", help="DDP 通信后端，GPU 训练通常使用 nccl")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ddp_ctx = setup_distributed(args)
    seed_everything(args.seed, ddp_ctx.rank)

    if args.use_swanlab:
        try:
            swanlab = importlib.import_module("swanlab")
        except Exception as exc:
            raise RuntimeError("已传入 --use_swanlab，但 swanlab 导入失败，请检查安装和运行环境。") from exc
        if ddp_ctx.is_main_process:
            swanlab.init(project="Happy-LLM", experiment_name="Pretrain-215M-DDP", config=vars(args))

    lm_config = ModelConfig(dim=1024, n_layers=18)
    max_seq_len = lm_config.max_seq_len
    args.save_dir = args.out_dir
    os.makedirs(args.save_dir, exist_ok=True)

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    ptdtype = dtype_map[args.dtype]
    if ddp_ctx.device.type == "cuda" and args.dtype != "float32":
        autocast_ctx = torch.cuda.amp.autocast(dtype=ptdtype)
    else:
        autocast_ctx = nullcontext()

    model, tokenizer = init_model()

    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=max_seq_len)
    train_loader, train_sampler = create_train_dataloader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        ddp_ctx=ddp_ctx,
        drop_last=False,
    )

    # bfloat16 通常不需要 GradScaler；float16 使用它可以降低梯度下溢风险。
    scaler = torch.cuda.amp.GradScaler(enabled=(ddp_ctx.device.type == "cuda" and args.dtype == "float16"))
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    iter_per_epoch = len(train_loader)
    try:
        for epoch in range(args.epochs):
            train_epoch(epoch)

        final_ckp = os.path.join(
            args.save_dir,
            f"pretrain_{lm_config.dim}_{lm_config.n_layers}_{lm_config.vocab_size}.pth",
        )
        save_checkpoint(model, final_ckp, ddp_ctx)
        logger(f"训练完成，模型已保存到：{final_ckp}")
    finally:
        cleanup_distributed(ddp_ctx)
