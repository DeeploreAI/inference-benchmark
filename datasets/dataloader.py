from pathlib import Path

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def data_loader(args):
    # Directory.
    root = Path(args.root)
    train_dir = root / "train"
    val_dir = root / "val"

    # Image pre-processing.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    basic_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    if args.augmentation:
        aug_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        train_set = datasets.ImageFolder(train_dir, aug_transforms)

        # Save index to class dictionary.
        # import yaml
        # idx_to_cls_dict = {v: k for k, v in train_set.class_to_idx.items()}
        # with open("/home/ziliang/Projects/Marine Datasets/butterflyfishes_cls/idx_to_class.yaml", "w") as f:
        #     yaml.dump(idx_to_cls_dict, f, indent=4)
    else:
        train_set = datasets.ImageFolder(train_dir, basic_transforms)
    val_set = datasets.ImageFolder(val_dir, basic_transforms)

    # Loader.
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        sampler=None,
        drop_last=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False
    )
    return train_loader, val_loader