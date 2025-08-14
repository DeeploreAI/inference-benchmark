import argparse
import datetime
from typing import Optional, List
import yaml
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os

from models.utils import Model
from datasets.dataloader import data_loader
from models.metrics import topk_accuracy


MODEL_NAMES = ["resnet50", "resnet101", "vgg16", "vgg19"]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="./configs/butterflyfishes.yaml")
    parser.add_argument("--resume_version", type=int, default=None)
    
    # Parse the config file path first.
    args, unknown = parser.parse_known_args(argv)
    
    # Load YAML config and merge with args.
    args = merge_yaml_with_args(args.cfg, args)
    return args


def merge_yaml_with_args(config_path: str, args: argparse.Namespace) -> argparse.Namespace:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            print(f"Not a valid config file: {config_path}")
            return args
        
        # Convert args to dict.
        args_dict = vars(args)
        
        # Merge config with args.
        for key, value_dict in config.items():
            if type(value_dict) is dict:
                for k, v in value_dict.items():
                    args_dict[k] = v
        
        # Convert back to Namespace.
        return argparse.Namespace(**args_dict)
        
    except FileNotFoundError:
        print(f"Warning: Config file {config_path} not found. Using default arguments.")
        return args
    except yaml.YAMLError as e:
        print(f"Warning: Error parsing YAML file {config_path}: {e}. Using default arguments.")
        return args


def setup_logging(args):
    if args.expt_name:
        expt_name = args.expt_name
    else:
        expt_name = f"{args.model_name}_{datetime.datetime.now().strftime('%m%d_%H%M')}"
    if args.resume_version:
        # Resume training.
        log_dir = Path("./logs") / f"{expt_name}" / f"version_{args.resume_version + 1}"
    else:
        # First time training.
        log_dir = Path("./logs") / f"{expt_name}" / "version_0"
    os.makedirs(log_dir, exist_ok=True)
    logger = SummaryWriter(log_dir=log_dir)
    return logger, log_dir


def optimizer_scheduler(args, model, num_iters):
    # Initialize optimizer.
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Create warmup scheduler and cosine annealing scheduler.
    warmup_iters_ratio = 0.1
    warmup_iters = int(num_iters * warmup_iters_ratio)
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-6,
        end_factor=1.0,
        total_iters=warmup_iters,
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_iters - warmup_iters,
        eta_min=1e-6,
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        [warmup_scheduler, cosine_scheduler],
        milestones=[warmup_iters],
    )
    return optimizer, scheduler


def save_ckpt(model_state, ckpt_path: Path, is_best=False):
    torch.save(model_state, ckpt_path)
    if is_best:
        shutil.copyfile(ckpt_path, ckpt_path.parent / "best.pth")


def load_ckpt(ckpt_path: Path, model, optimizer):
    if not ckpt_path.exists():
        print(f"Checkpoint file {ckpt_path} not found!")
        return None, 0

    print("=" * 60)
    print(f"\nResume Training from {ckpt_path}\n")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # Load model state dict.
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Load optimizer state if provided.
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Get training info
    start_epoch = checkpoint.get('epoch', 0)
    return optimizer, start_epoch


def load_pretrained_ckpt(model, pretrained_ckpt: Path):
    pretrained_dict = torch.load(pretrained_ckpt)
    model_dict = model.state_dict()

    # Check the model dict keys
    for v1, v2 in zip(pretrained_dict.values(), model_dict.values()):
        if v1.shape == v2.shape:
            continue
        else:
            raise ValueError("Pretrained params and current params do not have the same shape.")
        # param_name = k_1.split('.')[-1]
        # if "conv" in k_1.lower():
        #     layer_name = "conv"
        # elif "bn" in k_1.lower():
        #     layer_name = "bn"
        # elif

    # Rename the pretrained dict keys
    pretrained_dict = {k2: v1 for k2, v1 in zip(model_dict.keys(), pretrained_dict.values())}

    # Overwrite model dict
    model_dict.update(pretrained_dict)

    # Load the pretrained model dict
    model.load_state_dict(model_dict)
    return model


def get_latest_ckpt(ckpt_dir: Path):
    if not ckpt_dir.exists():
        return None
    
    # Find all checkpoint files.
    ckpt_files = list(ckpt_dir.glob("epoch_*.pth"))
    if not ckpt_files:
        return None
    
    # Sort by epoch number and return the latest.
    ckpt_files.sort(key=lambda x: int(x.stem.split('_')[1]))
    return ckpt_files[-1]


def save_training_config(args, log_dir: Path):
    """Save version information and config for reproducibility."""
    version_info = {
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_name': args.model_name,
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_size': getattr(args, 'batch_size', 'N/A'),
        'resume_version': getattr(args, 'resume_version', None),
        'expt_name': getattr(args, 'expt_name', None)
    }

    # Save full config
    config_file = log_dir / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    print(f"Training config saved to {config_file}.")


def train(train_loader, model, criterion, optimizer, scheduler, epoch, logger, global_iter, verbose_iters: int = 50):
    # Switch mode.
    model.train()
    
    running_loss = 0.0
    running_top1_acc = 0.0
    running_top3_acc = 0.0
    running_top5_acc = 0.0
    num_batches = len(train_loader)

    for batch_idx, (input, target) in enumerate(train_loader):
        # Forward propagation.
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(input)
        loss = criterion(output, target)

        # Metrics.
        topk_acc = topk_accuracy(output, target, topk=(1, 3, 5))
        top1_acc, top3_acc, top5_acc = [acc.item() for acc in topk_acc]

        # Backward propagation, update optimizer and scheduler.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Update running metrics
        running_loss += loss.item()
        running_top1_acc += top1_acc
        running_top3_acc += top3_acc
        running_top5_acc += top5_acc
        
        # Log metrics for each iteration
        current_lr = optimizer.param_groups[0]['lr']
        logger.add_scalar('Iter/Loss', loss.item(), global_iter)
        logger.add_scalar('Iter/Top1_Accuracy', top1_acc, global_iter)
        logger.add_scalar('Iter/Top3_Accuracy', top3_acc, global_iter)
        logger.add_scalar('Iter/Top5_Accuracy', top5_acc, global_iter)
        logger.add_scalar('Iter/Learning_Rate', current_lr, global_iter)
        
        global_iter += 1
        
        # Print progress every 100 iterations
        if verbose_iters and (batch_idx+1) % verbose_iters == 0:
            print(f'Epoch [{epoch}][{batch_idx+1}/{num_batches}] '
                  f'Loss: {loss.item():.4f} '
                  f'Top1: {top1_acc:.2f}% '
                  f'Top3: {top3_acc:.2f}% '
                  f'Top5: {top5_acc:.2f}% '
                  f'LR: {current_lr:.6f}')
    
    # Log epoch-level metrics
    avg_loss = running_loss / num_batches
    avg_top1_acc = running_top1_acc / num_batches
    avg_top3_acc = running_top3_acc / num_batches
    avg_top5_acc = running_top5_acc / num_batches
    logger.add_scalar('Epoch/Train_Loss', avg_loss, epoch)
    logger.add_scalar('Epoch/Train_Top1_Accuracy', avg_top1_acc, epoch)
    logger.add_scalar('Epoch/Train_Top3_Accuracy', avg_top3_acc, epoch)
    logger.add_scalar('Epoch/Train_Top5_Accuracy', avg_top5_acc, epoch)
    
    return global_iter


def validate(val_loader, model, criterion, epoch, logger, verbose_iters: int = 10):
    # Switch mode.
    model.eval()

    running_loss = 0.0
    running_top1_acc = 0.0
    running_top3_acc = 0.0
    running_top5_acc = 0.0
    num_batches = len(val_loader)

    for batch_idx, (input, target) in enumerate(val_loader):
        # Forward propagation.
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

            # Metrics.
            topk_acc = topk_accuracy(output, target, topk=(1, 3, 5))
            top1_acc, top3_acc, top5_acc = [acc.item() for acc in topk_acc]

            # Update running metrics.
            running_loss += loss.item()
            running_top1_acc += top1_acc
            running_top3_acc += top3_acc
            running_top5_acc += top5_acc

            # Print progress every several iterations.
            if verbose_iters and (batch_idx + 1) % verbose_iters == 0:
                print(f'Val Loss: {loss.item():.4f} '
                      f'Val Top1: {top1_acc:.2f}% '
                      f'Val Top3: {top3_acc:.2f}% '
                      f'Val Top5: {top5_acc:.2f}%')

    # Log epoch-level metrics
    avg_loss = running_loss / num_batches
    avg_top1_acc = running_top1_acc / num_batches
    avg_top3_acc = running_top3_acc / num_batches
    avg_top5_acc = running_top5_acc / num_batches
    logger.add_scalar('Val/Loss', avg_loss, epoch)
    logger.add_scalar('Val/Top1_Accuracy', avg_top1_acc, epoch)
    logger.add_scalar('Val/Top3_Accuracy', avg_top3_acc, epoch)
    logger.add_scalar('Val/Top5_Accuracy', avg_top5_acc, epoch)
    return avg_loss, avg_top1_acc, avg_top3_acc, avg_top5_acc


def to_device(nn_module_list: List):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_module_list = []
    for module in nn_module_list:
        device_module_list.append(module.to(device))
    return device_module_list


def main(args) -> None:
    # Setup logging.
    logger, log_dir = setup_logging(args)
    
    # Save training config for resuming.
    save_training_config(args, log_dir)
    
    # Load model.
    if args.model_name in MODEL_NAMES:
        with open(f"./configs/models/{args.model_name}.yaml", "r") as f:
            model_cfg = yaml.safe_load(f)
        model = Model(model_cfg)
    else:
        print("Not a supported model.")
        return

    # Load datasets.
    train_loader, val_loader = data_loader(args)
    num_iters = len(train_loader) * args.epochs

    # Define optimizer, scheduler and loss.
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = optimizer_scheduler(args, model, num_iters)

    # Training and validating.
    model, criterion = to_device([model, criterion])
    global_iter = 0
    best_top1_acc = 0.0

    # Handle resume training.
    if args.resume_version:
        resume_ckpts_dir = log_dir.parent / f"version_{args.resume_version}" / "ckpts"
        ckpt_path = get_latest_ckpt(resume_ckpts_dir)
        if ckpt_path:
            optimizer, start_epoch = load_ckpt(ckpt_path, model, optimizer)
            global_iter = start_epoch * len(train_loader)  # Resume from the last global_iter
            print(f"Resuming training from epoch {start_epoch}, best top1 accuracy: {best_top1_acc:.2f}%")
        else:
            print(f"No checkpoint found for resume_version {args.resume_version}. Starting from epoch 1.")
            start_epoch = 1
    else:
        # First time training.
        if args.pretrained:
            pretrained_ckpt_path = Path("./pretrained") / f"{args.model_name}_pretrained.pth"
            model = load_pretrained_ckpt(model, pretrained_ckpt_path)
        start_epoch = 1

    # Start training.
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        global_iter = train(train_loader, model, criterion, optimizer, scheduler, epoch, logger, global_iter, verbose_iters=50)

        if args.validate_epoch:
            if epoch % args.validate_epoch == 0:
                # Validate.
                loss, top1_acc, top3_acc, top5_acc = validate(val_loader, model, criterion, epoch, logger, verbose_iters=10)

                # Save checkpoint file.
                model_state = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_top1_acc": top1_acc,
                    "val_top3_acc": top3_acc,
                    "val_top5_acc": top5_acc,
                }
                ckpt_file = log_dir / "ckpts" / f"epoch_{epoch}.pth"
                ckpt_file.parent.mkdir(parents=True, exist_ok=True)
                if top1_acc > best_top1_acc:
                    is_best = True
                    best_top1_acc = top1_acc
                    print(f"Best epoch at epoch {epoch}, top1_acc: {best_top1_acc:.2f}%")
                else:
                    is_best = False
                save_ckpt(model_state, ckpt_file, is_best=is_best)
    
    # Close logger
    logger.close()
    print("Training completed. Logs saved to:", logger.log_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)