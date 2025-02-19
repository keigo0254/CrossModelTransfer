import os

import torch


def setup_ddp(rank: int, world_size: int, port: int = 12357) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # Initialize process group
    torch.distributed.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)
    torch.distributed.barrier()


def cleanup_ddp() -> None:
    torch.distributed.destroy_process_group()


def is_main_process() -> bool:
    return torch.distributed.get_rank() == 0


def distribute_loader(loader: torch.utils.data.DataLoader) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        loader.dataset,
        batch_size=loader.batch_size // torch.distributed.get_world_size(),
        sampler=torch.utils.data.distributed.DistributedSampler(
            loader.dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=True
        ),
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
    )
