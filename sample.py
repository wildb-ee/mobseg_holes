import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import pathlib

from utils.sdataset import preprocess_raw
from utils.model import HoleNet

def predict_img(
    net,
    impil: Image.Image,
    device,
    dims: tuple[int, int] = (512, 512),
    out_threshold: float = 0.5,
):
    net.eval()

    img = preprocess_raw(impil, dims)
    img = img.unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
        memory_format=torch.channels_last,
    )

    with torch.inference_mode():
        output = net(img)

        output = F.interpolate(
            output,
            size=(impil.height, impil.width),
            mode="bilinear",
            align_corners=False,
        )

        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return (
        mask[0]
        .squeeze()
        .long()
        .cpu()
        .numpy()
    )


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    val_dir = pathlib.Path('val_imgs')
    in_files = [str(file) for file in val_dir.glob("*.jpg")]
    out_files = [str(file.with_stem(file.stem+"_new").with_suffix('.png')) for file in val_dir.glob("*.jpg")]

    net = HoleNet()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model...')
    logging.info(f'Using device {device}')

    net.to(device=device)
    # net = torch.compile(model=net) # after gpu move always
    
    checkpoint_dir = pathlib.Path("checkpoints")
    cp_path = max(
        checkpoint_dir.glob("checkpoint_epoch_*"),
        key=lambda p: int(p.stem.split("_")[-1])
    )
    state_dict = torch.load(cp_path, map_location=device)
    mask_values = [0,1]
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)
        if filename.endswith("_real.jpg"):
            img = ImageOps.invert(img)

        mask = predict_img(net=net,
                           impil=img,
                           out_threshold=0.5,
                           device=device)

        out_filename = out_files[i]
        result = mask_to_image(mask, mask_values)
        result.save(out_filename)
        logging.info(f'Mask saved to {out_filename}')