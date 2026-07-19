import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import torch
import pathlib
import random
from PIL import Image

class SDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        image_path = str(self.image_paths[idx])
        mask_path = str(self.mask_paths[idx])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask

def preprocess_raw(image_pil: Image, device, dims: tuple = (512,512)):
    inference_transform = A.Compose([
        A.Resize(dims[0], dims[1]),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        A.ToTensorV2(),
    ])

    image = np.array(
        image_pil
    )
    image = inference_transform(image=image)["image"]

    return image

def load_data(dir: str, batch_size: int, dims: tuple, split: float):
    train_transform = A.Compose([
        A.RandomCrop(height=dims[0], width=dims[1], p=1.0),
        A.SquareSymmetry(p=1.0), 
        A.RandomBrightnessContrast(p=0.3), # only image 
        A.GaussNoise(std_range=(0.1, 0.2), p=0.4), # only image
        A.GaussianBlur(p=0.1), # only image
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # imagenet defaults
        A.ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.CenterCrop(height=dims[0], width=dims[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.ToTensorV2(),
    ])
    
    mask_dir =pathlib.Path(dir) / "masks"
    image_dir = pathlib.Path(dir) / "images"

    image_paths = [
        file for file in image_dir.iterdir() 
        if file.suffix.lower() in ('.png', '.jpg', '.jpeg')
    ]
    mask_paths =[
        mask_dir / image_path.with_suffix(".png").name
        for image_path in image_paths
    ]

    assert all(path.exists() for path in mask_paths)

    shuffled_inds= random.sample(range(len(image_paths)), len(image_paths))
    shuffled_imgpaths = [image_paths[i] for i in shuffled_inds]
    shuffled_maskpaths = [mask_paths[i] for i in shuffled_inds]

    split_idx = int(split*len(shuffled_imgpaths))

    train_dataset = SDataset(
        image_paths=shuffled_imgpaths[:split_idx],
        mask_paths=shuffled_maskpaths[:split_idx],
        transform=train_transform
    )

    val_dataset = SDataset(
        image_paths=shuffled_imgpaths[split_idx:],
        mask_paths=shuffled_maskpaths[split_idx:],
        transform=val_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, split_idx, len(shuffled_imgpaths)-split_idx 


def overlay_mask(image, mask, alpha=0.5, color=(0, 1, 0)): # green overlay
    mask_overlay = np.zeros_like(image, dtype=np.uint8)
    mask_overlay[mask > 0] = (np.array(color) * 255).astype(np.uint8)

    overlayed_image = cv2.addWeighted(image, 1, mask_overlay, alpha, 0)
    return overlayed_image


def visualize_segmentation(dataset, idx=0, samples=3):
    if isinstance(dataset.transform, A.Compose):
        vis_transform_list = [
            t for t in dataset.transform
            if not isinstance(t, (A.Normalize, A.ToTensorV2))
        ]
        vis_transform = A.Compose(vis_transform_list)
    else:
        print("warning: could not automatically strip Normalize/ToTensor for visualization.")
        vis_transform = dataset.transform

    figure, ax = plt.subplots(samples + 1, 2, figsize=(8, 4 * (samples + 1)))

    original_transform = dataset.transform
    dataset.transform = None # temp disable for raw data access
    image, mask = dataset[idx]
    dataset.transform = original_transform 

    ax[0, 0].imshow(image)
    ax[0, 0].set_title("Original Image")
    ax[0, 0].axis("off")
    ax[0, 1].imshow(overlay_mask(image, mask)) 
    ax[0, 1].set_title("Original Overlay")

    for i in range(samples):
        if vis_transform:
            augmented = vis_transform(image=image, mask=mask)
            aug_image = augmented['image']
            aug_mask = augmented['mask']
        else:
            aug_image, aug_mask = image, mask 

        ax[i + 1, 0].imshow(aug_image)
        ax[i + 1, 0].set_title(f"Augmented Image {i+1}")
        ax[i + 1, 0].axis("off")

        ax[i+1, 1].imshow(overlay_mask(aug_image, aug_mask))
        ax[i+1, 1].set_title(f"Augmented Overlay {i+1}")

    plt.tight_layout()
    plt.show()