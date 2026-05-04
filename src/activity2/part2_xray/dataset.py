import os

import kagglehub
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset, random_split
from torchvision.datasets import ImageFolder


# ImageNet stats — required when using pretrained weights so the activations
# stay close to the distribution they were trained on.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

KAGGLE_DATASET = "paultimothymooney/chest-xray-pneumonia"


def download_xray_dataset() -> str:
    """
    Download (or reuse the cache of) the Chest X-Ray dataset from Kaggle and
    return the path to the `chest_xray` folder containing `train/val/test`.

    kagglehub stores it under `~/.cache/kagglehub/...` and returns the path
    of the downloaded version. The internal layout of the ZIP is:
        <root>/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg
    """
    root = kagglehub.dataset_download(KAGGLE_DATASET)
    chest_xray_dir = os.path.join(root, "chest_xray")
    if not os.path.isdir(os.path.join(chest_xray_dir, "train")):
        raise FileNotFoundError(
            f"Unexpected structure at {chest_xray_dir}. "
            f"Expected train/val/test subdirectories."
        )
    return chest_xray_dir


def get_dataloaders(data_dir: str | None = None, batch_size: int = 32,
                    img_size: int = 224, num_workers: int = 2,
                    val_split: float = 0.1, seed: int = 42):
    """
    Load the Chest X-Ray dataset (ImageFolder layout).

    The original `val/` split has only 16 images — useless for early stopping.
    By default we carve `val_split` (10%) of `train/` as the new validation
    set. The 16 original val images are appended to the test set.

    If `data_dir` is None, the dataset is downloaded via kagglehub.

    Returns: (train_loader, val_loader, test_loader, classes)
    """
    if data_dir is None:
        data_dir = download_xray_dataset()

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset not found at {data_dir}.")

    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    transform_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # Load train twice (with and without augmentation) so the validation
    # subset uses only deterministic transforms.
    train_full_aug  = ImageFolder(os.path.join(data_dir, "train"), transform=transform_train)
    train_full_eval = ImageFolder(os.path.join(data_dir, "train"), transform=transform_eval)
    orig_val_set    = ImageFolder(os.path.join(data_dir, "val"),   transform=transform_eval)
    test_set        = ImageFolder(os.path.join(data_dir, "test"),  transform=transform_eval)

    n_total = len(train_full_aug)
    n_val   = int(n_total * val_split)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(
        range(n_total), [n_train, n_val], generator=generator,
    )

    train_set = Subset(train_full_aug,  list(train_indices))
    val_set   = Subset(train_full_eval, list(val_indices))

    # Append the 16 original val images to test so they are not wasted.
    test_set = ConcatDataset([test_set, orig_val_set])

    # pin_memory only helps on CUDA (unsupported on MPS, irrelevant on CPU).
    pin_memory = torch.cuda.is_available()
    persistent = num_workers > 0
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent,
    )

    return train_loader, val_loader, test_loader, train_full_aug.classes
