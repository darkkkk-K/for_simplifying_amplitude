"""RoPE contrastive pretraining, AdamW and warmup/cosine schedule."""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os,time,pickle,math,random
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from data.tokenizer import collate_fn
from data.distributed import AdvancedDistributedDataset
from training.evaluator import comprehensive_evaluate
from training.loss import NTXentloss
from models.encoder import ClusteringMachine1_RoPE
def setup_distributed():
    """初始化分布式训练环境 - 修复版"""
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    print(f'🔧 Rank {rank}: 环境变量 - RANK={rank}, LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}')
    if world_size == 1:
        print(f'⚠️  Rank {rank}: 单GPU模式')
        return (0, 1, 0)
    if not torch.cuda.is_available():
        print(f'❌ Rank {rank}: CUDA不可用，无法进行分布式训练')
        return (0, 1, 0)
    try:
        torch.cuda.set_device(local_rank)
        print(f'✅ Rank {rank}: 设置GPU设备 {local_rank}')
    except Exception as e:
        print(f'❌ Rank {rank}: 设置GPU设备失败: {e}')
        return (0, 1, 0)
    try:
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        master_port = os.environ.get('MASTER_PORT', '12355')
        print(f'🔗 Rank {rank}: 连接到主节点 {master_addr}:{master_port}')
        dist.init_process_group(backend='nccl', init_method=f'tcp://{master_addr}:{master_port}', rank=rank, world_size=world_size)
        print(f'✅ Rank {rank}: 分布式初始化成功')
        return (rank, world_size, local_rank)
    except Exception as e:
        print(f'❌ Rank {rank}: 分布式初始化失败: {e}')
        return (0, 1, 0)

def cleanup_distributed(rank):
    """清理分布式训练环境"""
    if not dist.is_initialized():
        print(f'Rank {rank}: 分布式环境未初始化，无需清理')
        return
    try:
        print(f'Rank {rank}: 等待所有进程准备清理...')
        dist.barrier()
        print(f'Rank {rank}: 开始销毁进程组')
        dist.destroy_process_group()
        print(f'Rank {rank}: 进程组销毁完成')
    except Exception as e:
        print(f'Rank {rank}: 清理分布式环境时出错: {e}')

def monitor_gpus(rank, local_rank):
    """
    监控当前进程的GPU状态
    """
    if not torch.cuda.is_available():
        print(f'Rank {rank}: CUDA不可用')
        return
    device = torch.device(f'cuda:{local_rank}')
    memory_allocated = torch.cuda.memory_allocated(device) / 1024 ** 3
    memory_reserved = torch.cuda.memory_reserved(device) / 1024 ** 3
    memory_max = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
    print(f'Rank {rank} (Local Rank {local_rank}): GPU {local_rank} - 已分配: {memory_allocated:.2f} GB, 保留: {memory_reserved:.2f} GB, 总计: {memory_max:.2f} GB')

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
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(progress, 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        decayed_lr = (1 - min_lr_ratio) * cosine_decay + min_lr_ratio
        return decayed_lr
    return LambdaLR(optimizer, lr_lambda)

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

def train_model(config, if_generate_data: bool=True):
    """
    训练函数，model_name需要加pth后缀
    """
    rank, world_size, local_rank = setup_distributed()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        print(f'✅ Rank {rank}: 已设置设备为 cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    seed = 42 + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if rank == 0:
        os.makedirs(config['output_dir'], exist_ok=True)
        print(f'训练设备：{device}, 进程排名：{rank}/{world_size - 1}')
        print(f'总进程数：{world_size}')
    if if_generate_data and rank == 0:
        raise ValueError('Generate encoder data with scripts/generate_data.py first')
    if dist.is_initialized():
        dist.barrier()
    src_train = AdvancedDistributedDataset(data_dir='./datasets/pretrain/train_data', shuffle_files=False, max_files_in_memory=4)
    src_val = AdvancedDistributedDataset(data_dir='./datasets/pretrain/val_data', shuffle_files=False, max_files_in_memory=4)
    train_sampler = DistributedSampler(src_train, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(src_val, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(src_train, batch_size=config['batch_size'], sampler=train_sampler, collate_fn=collate_fn, num_workers=config['num_workers'], pin_memory=True, prefetch_factor=4, persistent_workers=True)
    val_loader = DataLoader(src_val, batch_size=config['batch_size'], sampler=val_sampler, collate_fn=collate_fn, num_workers=config['num_workers'], pin_memory=True, prefetch_factor=4, persistent_workers=True)
    if rank == 0:
        num_train_samples = len(src_train)
        num_val_samples = len(src_val)
        print(f'训练样本：{num_train_samples}，验证样本：{num_val_samples}')
        print(f'每个进程的批次数量：训练集 {len(train_loader)}，验证集 {len(val_loader)}')
    if config['model_type'] == 'RoPE':
        model1 = ClusteringMachine1_RoPE(vocab_size=config['vocab_size'], emb_dim=config['emb_dim'], numhead=config['numhead'], numenc1=config['numenc'], fnn_dim=config['fnn_dim'], dropout=config['dropout'], RoPE_mode=config['RoPE_mode']).to(device)
    else:
        raise ValueError('Use the RoPE encoder configuration')
    if dist.is_initialized():
        model1 = DDP(model1, device_ids=[local_rank], output_device=local_rank)
    if rank == 0:
        total_params = sum((p.numel() for p in model1.parameters()))
        print(f'模型参量总数：{total_params}')
    scaler = torch.amp.GradScaler() if config.get('use_amp', True) else None
    batches_per_epoch = len(train_loader)
    total_steps = config['num_epoch'] * (batches_per_epoch // config['gradient_accumulation_steps'])
    warmup_steps = int(0.12 * total_steps)
    criterion = NTXentloss(temperature=config['loss_temp'])
    optimizer = optim.AdamW(model1.parameters(), lr=config['learning_rate'], betas=(0.9, 0.998), eps=1e-08, weight_decay=0.01)
    scheduler = create_manual_scheduler(optimizer=optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_lr_ratio=0.1)
    best_val_loss = float('inf')
    train_history = {'train_loss': [], 'val_loss': [], 'diff_inter': [], 'spherical_anergy': [], 'epoch_train_loss': []}
    if rank == 0:
        print('\n开始训练')
    start_time = time.time()
    for epoch in range(config['num_epoch']):
        train_sampler.set_epoch(epoch)
        model1.train()
        epoch_train_loss = 0.0
        for batch_idx, src_in in enumerate(train_loader):
            src_in = src_in.to(device, non_blocking=True)
            if scaler:
                with torch.amp.autocast(device_type='cuda'):
                    output = model1(src_in)
                    if rank == 0 and (batch_idx + 1) % config['log_interval'] == 0:
                        monitor_gpus(rank, local_rank)
                    loss = criterion(output)
                scaler.scale(loss / config['gradient_accumulation_steps']).backward()
            else:
                output = model1(src_in)
                if rank == 0 and (batch_idx + 1) % config['log_interval'] == 0:
                    monitor_gpus(rank, local_rank)
                loss = criterion(output)
                (loss / config['gradient_accumulation_steps']).backward()
            reduced_loss = reduce_value(loss, average=True)
            if 2 * (batch_idx + 1) % config['log_interval'] == 0:
                print(f'Batch {batch_idx} in rank{rank}, Average Loss: {loss:.4f}')
            if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
                clip_grad_norm_(model1.parameters(), config['max_grad_norm'])
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            epoch_train_loss += reduced_loss
            if (batch_idx + 1) % config['log_interval'] == 0:
                diff_per_interval, spherical_energy_perinterval = comprehensive_evaluate(src=src_in, output=output, device=device)
                avg_loss = epoch_train_loss / (batch_idx + 1)
                train_history['train_loss'].append(reduced_loss)
                train_history['diff_inter'].append(diff_per_interval)
                train_history['spherical_anergy'].append(spherical_energy_perinterval)
                train_history['epoch_train_loss'].append(avg_loss)
                if rank == 0:
                    print(f"Epoch: {epoch + 1}/{config['num_epoch']}")
                    print(f'Batch: {batch_idx + 1}/{len(train_loader)}')
                    print(f'Average loss: {avg_loss:.4f}')
                    print(f'当前凝聚度和表面能：{diff_per_interval:.8f}, {spherical_energy_perinterval:.4f}')
                    if dist.is_initialized():
                        print(f'当前学习率: {scheduler.get_last_lr()[0]:.8f}')
            if (batch_idx + 1) % (100 * config['log_interval']) == 0 and rank == 0:
                temp_model_path = os.path.join(config['output_dir'], config['temp_model_name'])
                checkpoint = {'epoch': epoch, 'batch_idx': batch_idx, 'model_state_dict': model1.module.state_dict() if hasattr(model1, 'module') else model1.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict() if scheduler else None, 'scaler_state_dict': scaler.state_dict() if scaler else None, 'best_val_loss': best_val_loss, 'train_history': train_history}
                torch.save(checkpoint, temp_model_path)
                print(f'✅ 临时检查点已保存: {temp_model_path}')
            if (batch_idx + 1) % (100 * config['log_interval']) == 0:
                torch.cuda.empty_cache()
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
        train_history['val_loss'].append(avg_val_loss)
        if rank == 0:
            print(f"\nEpoch: {epoch + 1}/{config['num_epoch']}")
            print(f'Train loss: {avg_train_loss:.4f}')
            print(f'Val loss: {avg_val_loss:.4f}')
            print(f'Internal diff: {diff_inter_avg:.8f}')
            print(f'Spherical energy: {spherical_energy_avg:.4f}')
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_path = os.path.join(config['output_dir'], config['best_model_name'])
                if hasattr(model1, 'module'):
                    torch.save(model1.module.state_dict(), model_path)
                else:
                    torch.save(model1.state_dict(), model_path)
                print(f'保存最佳模型到 {model_path}')
    if rank == 0:
        final_model_path = os.path.join(config['output_dir'], config['final_model_name'])
        if hasattr(model1, 'module'):
            torch.save(model1.module.state_dict(), final_model_path)
        else:
            torch.save(model1.state_dict(), final_model_path)
        log_dir = os.path.join(config['output_dir'], 'log_dir')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, config['log_file_name'])
        with open(log_path, 'wb') as f:
            pickle.dump(train_history, f)
        training_time = time.time() - start_time
        print(f'训练完成! 总时间: {training_time // 60:.0f}分 {training_time % 60:.0f}秒')
    cleanup_distributed(rank)
    return train_history
