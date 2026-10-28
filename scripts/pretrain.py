import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from random import shuffle
import os
import time
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
import torch.multiprocessing as mp
from data.tokenizer import tokenize
from data.dataset import MyDataset, collate_fn
from data.generation import generate_distributed_dataset
# from data.generation import data_gen_for_pre
from training.evaluator import comprehensive_evaluate
from models.encoder import ClusteringMachine1, ClusteringMachine1_RoPE
from training.loss import NTXentloss
# from data.generation import generate_distributed_dataset
from data.distributed import AdvancedDistributedDataset
import argparse
import pickle
import math
import json
import random

def setup_distributed():
    """初始化分布式训练环境 - 修复版"""
    # 获取环境变量
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    print(f"🔧 Rank {rank}: 环境变量 - RANK={rank}, LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}")

    # 如果world_size为1，则不需要分布式训练
    if world_size == 1:
        print(f"⚠️  Rank {rank}: 单GPU模式")
        return 0, 1, 0

    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print(f"❌ Rank {rank}: CUDA不可用，无法进行分布式训练")
        return 0, 1, 0

    # 设置GPU设备
    try:
        torch.cuda.set_device(local_rank)
        print(f"✅ Rank {rank}: 设置GPU设备 {local_rank}")
    except Exception as e:
        print(f"❌ Rank {rank}: 设置GPU设备失败: {e}")
        return 0, 1, 0

    # 初始化分布式进程组
    try:
        # 设置主节点地址和端口
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        master_port = os.environ.get('MASTER_PORT', '12355')

        print(f"🔗 Rank {rank}: 连接到主节点 {master_addr}:{master_port}")

        # 初始化进程组
        dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://{master_addr}:{master_port}',
            rank=rank,
            world_size=world_size
        )

        print(f"✅ Rank {rank}: 分布式初始化成功")
        return rank, world_size, local_rank

    except Exception as e:
        print(f"❌ Rank {rank}: 分布式初始化失败: {e}")
        return 0, 1, 0

def cleanup_distributed(rank):
    """清理分布式训练环境"""
    if not dist.is_initialized():
        print(f"Rank {rank}: 分布式环境未初始化，无需清理")
        return

    try:
        # 同步所有进程，确保都准备好清理
        print(f"Rank {rank}: 等待所有进程准备清理...")
        dist.barrier()

        # 执行清理
        print(f"Rank {rank}: 开始销毁进程组")
        dist.destroy_process_group()
        print(f"Rank {rank}: 进程组销毁完成")

    except Exception as e:
        print(f"Rank {rank}: 清理分布式环境时出错: {e}")
        # 即使出错也要继续，避免阻塞

def save_checkpoint(model, optimizer, scheduler, epoch, loss, config, path, use_ddp=False):
    """保存检查点"""
    if use_ddp:
        model_state = model.module.state_dict()  # 保存DDP包装前的模型
    else:
        model_state = model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'config': config
    }
    torch.save(checkpoint, path)

def monitor_gpus(rank, local_rank):
    """
    监控当前进程的GPU状态
    """
    if not torch.cuda.is_available():
        print(f"Rank {rank}: CUDA不可用")
        return

    # 当前进程使用的GPU
    device = torch.device(f'cuda:{local_rank}')
    memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
    memory_reserved = torch.cuda.memory_reserved(device) / 1024**3   # GB
    memory_max = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB

    print(f"Rank {rank} (Local Rank {local_rank}): GPU {local_rank} - 已分配: {memory_allocated:.2f} GB, 保留: {memory_reserved:.2f} GB, 总计: {memory_max:.2f} GB")

def create_optimizer(model, config):
    """创建优化器"""
    return optim.Adam(
        model.parameters(),
        lr = config['learning_rate'],
        betas = (0.9, 0.98),
        eps = 1e-9
    )

def create_manual_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    """
    创建手动学习率调度器

    Args:
        optimizer: 优化器实例
        warmup_steps: 预热步数
        total_steps: 总训练步数
        min_lr_ratio: 最小学习率与基础学习率的比例
    """

    def lr_lambda(current_step):
        # 线性预热阶段
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        # 余弦退火阶段
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(progress, 1.0)  # 确保不超过1

        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        decayed_lr = (1 - min_lr_ratio) * cosine_decay + min_lr_ratio

        return decayed_lr

    return LambdaLR(optimizer, lr_lambda)

def save_checkpoint(epoch, model, optimizer, scheduler, scaler, loss, config, rank):
    """保存检查点(只在rank 0执行)"""
    if rank != 0:
        return

    os.makedirs(config["output_dir"], exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'loss': loss,
    }

    checkpoint_path = os.path.join(config["output_dir"], f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"检查点已保存: {checkpoint_path}")

def reduce_value(value, average=True):
    """在分布式环境中规整值"""
    if not dist.is_initialized():
        return value

    if isinstance(value, torch.Tensor):
        value = value.clone().detach()
    else:
        value = torch.tensor(value, device=torch.cuda.current_device())

    dist.all_reduce(value, async_op=False)
    if average:
        value /= dist.get_world_size()

    return value.item() if value.numel() == 1 else value

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, device, rank):
        if not os.path.exists(checkpoint_path):
            return 0, float('inf'), {}

        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 加载模型状态
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

        # 加载优化器状态
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 加载调度器状态
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # 加载scaler状态
        if scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # 加载训练历史
        history = checkpoint.get('train_history', {
            'train_loss': [], 'val_loss': [], 'diff_inter': [], 'spherical_anergy':[]
        })

        print(f"Rank {rank}: 从检查点恢复 - epoch {checkpoint['epoch']}, batch {checkpoint.get('batch_idx', 0)}")

        return checkpoint['epoch'], checkpoint.get('best_val_loss', float('inf')), history

def train_model(config, if_generate_data:bool = True):
    '''
    训练函数，model_name需要加pth后缀
    '''
    # ========================
    # 1. 分布式设置与初始化
    # ========================
    rank, world_size, local_rank = setup_distributed()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        print(f"✅ Rank {rank}: 已设置设备为 cuda:{local_rank}")
    else:
        device = torch.device("cpu")


    ## 设置随机数种子保证可复现性，考虑rank避免所有进程生成相同数据
    seed = 42 + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 只在主进程上创建输出目录和打印信息
    if rank == 0:
        os.makedirs(config['output_dir'], exist_ok=True)
        print(f'训练设备：{device}, 进程排名：{rank}/{world_size-1}')
        print(f'总进程数：{world_size}')

    # ========================
    # 2. 数据准备
    # ========================
    # 只在主进程上生成数据
    if if_generate_data and rank == 0:
        generate_distributed_dataset(
            total_samples=config['num_samples'],
            samples_per_file=config['samples_per_file'],
            data_dir='./datasets/pretrain',
            temperature=config['temperature']
        )

    # 等待主进程完成数据生成
    if dist.is_initialized():
        dist.barrier()

    src_train = AdvancedDistributedDataset(
        data_dir='./datasets/pretrain/train_data',
        shuffle_files=False,
        max_files_in_memory=4
    )
    src_val = AdvancedDistributedDataset(
        data_dir='./datasets/pretrain/val_data',
        shuffle_files=False,
        max_files_in_memory=4
    )

    # 创建分布式采样器
    train_sampler = DistributedSampler(
        src_train,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    val_sampler = DistributedSampler(
        src_val,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    # 数据加载器实例
    train_loader = DataLoader(
        src_train,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True,  # 加速数据转移到GPU
        prefetch_factor=4,  # 增加预取
        persistent_workers=True
    )

    val_loader = DataLoader(
        src_val,
        batch_size=config['batch_size'],
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True,
        prefetch_factor=4,  # 增加预取
        persistent_workers=True
    )

    if rank == 0:
        num_train_samples = len(src_train)
        num_val_samples = len(src_val)
        print(f'训练样本：{num_train_samples}，验证样本：{num_val_samples}')
        print(f'每个进程的批次数量：训练集 {len(train_loader)}，验证集 {len(val_loader)}')

    # ========================
    # 3. 模型初始化
    # ========================
    ## 初始化模型：
    if config['model_type'] == 'RoPE':
        model1 = ClusteringMachine1_RoPE(
            vocab_size=config['vocab_size'],
            emb_dim=config['emb_dim'],
            numhead=config['numhead'],
            numenc1=config['numenc'],
            fnn_dim=config['fnn_dim'],
            dropout=config['dropout'],
            RoPE_mode=config['RoPE_mode']
        ).to(device)
    else:
        model1 = ClusteringMachine1(
            vocab_size=config['vocab_size'],
            emb_dim=config['emb_dim'],
            numhead=config['numhead'],
            numenc1=config['numenc'],
            fnn_dim=config['fnn_dim'],
            dropout=config['dropout']
        ).to(device)

    # 使用DistributedDataParallel包装模型
    if dist.is_initialized():
        model1 = DDP(model1, device_ids=[local_rank], output_device=local_rank)

    if rank == 0:
        total_params = sum(p.numel() for p in model1.parameters())
        print(f'模型参量总数：{total_params}')

    # ========================
    # 4. 训练准备
    # ========================

    scaler = torch.amp.GradScaler() if config.get('use_amp', True) else None

    batches_per_epoch = len(train_loader)
    total_steps = config['num_epoch']*(batches_per_epoch//config['gradient_accumulation_steps'])
    warmup_steps = int(0.12 * total_steps)

    # 损失函数
    criterion = NTXentloss(temperature=config['loss_temp'])

    # 使用Adam优化器
    optimizer = optim.AdamW(
        model1.parameters(),
        lr=config['learning_rate'],
        betas=(0.9, 0.998),
        eps=1e-8,
        weight_decay=0.01
    )

    # 学习率动态调度器
    scheduler = create_manual_scheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=0.1
    )

    # 跟踪训练状态
    best_val_loss = float('inf')
    train_history = {
        'train_loss': [],
        'val_loss': [],
        'diff_inter': [],
        'spherical_anergy':[],
        'epoch_train_loss':[]
    }

    # ========================
    # 5. 训练循环
    # ========================
    if rank == 0:
        print('\n开始训练')
    start_time = time.time()

    for epoch in range(config['num_epoch']):
        # 设置epoch给采样器，确保每个epoch的shuffle不同
        train_sampler.set_epoch(epoch)

        model1.train()
        epoch_train_loss = 0.0

        for batch_idx, src_in in enumerate(train_loader):
            src_in = src_in.to(device, non_blocking=True)

            # 使用自动混合精度
            if scaler:
                with torch.amp.autocast(device_type='cuda'):
                    output = model1(src_in)
                    if rank == 0 and (batch_idx + 1) % config['log_interval'] == 0:
                        monitor_gpus(rank, local_rank)
                    loss = criterion(output)
                scaler.scale(loss / config['gradient_accumulation_steps']).backward()  # 在这里除
            else:
                output = model1(src_in)
                if rank == 0 and (batch_idx + 1) % config['log_interval'] == 0:
                        monitor_gpus(rank, local_rank)
                loss = criterion(output)
                (loss / config['gradient_accumulation_steps']).backward()  # 在这里除

            # 计算损失

            # 将每个进程的损失收集起来（求平均）
            reduced_loss = reduce_value(loss, average=True)
            if (2*(batch_idx + 1)) % config['log_interval'] == 0:
                print(f"Batch {batch_idx} in rank{rank}, Average Loss: {loss:.4f}")

            # 梯度累积
            if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
                clip_grad_norm_(model1.parameters(), config['max_grad_norm'])
                if scaler:
                    scaler.step(optimizer)  # 先反缩放梯度，再执行优化步骤
                    scaler.update()         # 更新缩放因子
                else:
                    optimizer.step()        # 非混合精度情况下直接使用

                optimizer.zero_grad()
                scheduler.step()

            epoch_train_loss += reduced_loss

            # 只在主进程上打印进度
            if (batch_idx + 1) % config['log_interval'] == 0:
                diff_per_interval, spherical_energy_perinterval = comprehensive_evaluate(src=src_in, output=output, device=device)
                avg_loss = epoch_train_loss / (batch_idx + 1)
                train_history['train_loss'].append(reduced_loss)
                train_history['diff_inter'].append(diff_per_interval)
                train_history['spherical_anergy'].append(spherical_energy_perinterval)
                train_history['epoch_train_loss'].append(avg_loss)
                if rank == 0:
                    print(f'Epoch: {epoch + 1}/{config["num_epoch"]}')
                    print(f'Batch: {batch_idx + 1}/{len(train_loader)}')
                    print(f'Average loss: {avg_loss:.4f}')
                    print(f'当前凝聚度和表面能：{diff_per_interval:.8f}, {spherical_energy_perinterval:.4f}')
                    if dist.is_initialized():
                        print(f'当前学习率: {scheduler.get_last_lr()[0]:.8f}')
            if (batch_idx + 1)%(100*config['log_interval']) == 0 and rank == 0:
                temp_model_path = os.path.join(config['output_dir'], config['temp_model_name'])

                # 保存完整检查点，包括优化器状态等
                checkpoint = {
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'model_state_dict': model1.module.state_dict() if hasattr(model1, 'module') else model1.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                    'best_val_loss': best_val_loss,
                    'train_history': train_history
                     }

                torch.save(checkpoint, temp_model_path)
                print(f"✅ 临时检查点已保存: {temp_model_path}")
            if (batch_idx + 1)%(100*config['log_interval']) == 0:
                torch.cuda.empty_cache()


        # 验证阶段
        model1.eval()
        val_loss = 0.0
        diff_inter = 0.0
        spherical_energy = 0.0
        batch_num = 0

        with torch.no_grad():
            for src in val_loader:
                src = src.to(device, non_blocking=True)

                output = model1(src)
                loss = criterion(output)
                val_loss += loss.item()

                diff, energy = comprehensive_evaluate(src=src, output=output, device=device)
                diff_inter += diff
                spherical_energy += energy
                batch_num += 1

        # 在所有进程间同步指标

        if world_size > 1:
            avg_train_loss = reduce_value(epoch_train_loss / len(train_loader))
            avg_val_loss = reduce_value(val_loss / max(batch_num, 1))
            diff_inter_avg = reduce_value(diff_inter / max(batch_num, 1))
            spherical_energy_avg = reduce_value(spherical_energy / max(batch_num, 1))
        else:
            avg_train_loss = epoch_train_loss / len(train_loader)
            avg_val_loss = val_loss / max(batch_num, 1)
            diff_inter_avg = diff_inter / max(batch_num, 1)
            spherical_energy_avg = spherical_energy / max(batch_num, 1)

        # 只在主进程上记录和打印
        train_history['val_loss'].append(avg_val_loss)
        if rank == 0:
            print(f'\nEpoch: {epoch + 1}/{config["num_epoch"]}')
            print(f'Train loss: {avg_train_loss:.4f}')
            print(f'Val loss: {avg_val_loss:.4f}')
            print(f'Internal diff: {diff_inter_avg:.8f}')
            print(f'Spherical energy: {spherical_energy_avg:.4f}')

            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_path = os.path.join(config['output_dir'], config['best_model_name'])
                # 保存原始模型状态（去除DDP包装）
                if hasattr(model1, 'module'):
                    torch.save(model1.module.state_dict(), model_path)
                else:
                    torch.save(model1.state_dict(), model_path)
                print(f"保存最佳模型到 {model_path}")

    # 训练结束，保存最终模型
    if rank == 0:
        final_model_path = os.path.join(config["output_dir"], config['final_model_name'])
        if hasattr(model1, 'module'):
            torch.save(model1.module.state_dict(), final_model_path)
        else:
            torch.save(model1.state_dict(), final_model_path)

        log_dir = os.path.join(config['output_dir'], 'log_dir')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, config['log_file_name'])
        with open(log_path, 'wb') as f:
            pickle.dump(train_history, f)

        # 训练时间统计
        training_time = time.time() - start_time
        print(f"训练完成! 总时间: {training_time//60:.0f}分 {training_time%60:.0f}秒")

    # 清理分布式环境
    cleanup_distributed()

    return train_history