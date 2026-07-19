import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from tqdm import tqdm

from utils.eval import eval
from utils.sdataset import load_data
from utils.dloss import dloss
from utils.model import HoleNet
import os
from dotenv import load_dotenv

load_dotenv()
check_dir = Path(os.getenv("CHECKPOINT_DIR"))
seg_dir = Path(os.getenv("SEG_DIR"))

EPOCHS = 10
BS = 4
LOADMEM = None
LR = 1e-5
SPLIT = 0.8
AMP = True

# TODO freeze the encoder for several epochs and then use small lr (transfer learning); get the blender sythetic properly setup for REALLY random imgs (black spots but no holes, oil stains , dark reflections etc)
def train(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        split: float = 0.1,
        save: bool = True,
        amp: bool = False,
        weight_decay: float = 1e-8,
        gradient_clipping: float = 1.0,
        dims: tuple = (512,512)
):

    tloader, vloader, tsize, vsize = load_data(seg_dir, batch_size, dims, split)
    eval_step = (tsize // (5 * batch_size))
    assert eval_step>0

    logging.info(f'''Starting training:
        epochs:          {epochs}
        batch size:      {batch_size}
        learning rate:   {learning_rate}
        training size:   {tsize}
        validation size: {vsize}
        checkpoints:     {save}
        device:          {device.type}
        amp: {amp}
    ''')

    optimizer = optim.AdamW(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    glstep = 0

    for epoch in range(1, epochs + 1):
        model.train()
        with tqdm(total=tsize, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for images, true_masks in tloader:

                assert images.shape[1] == 3, \
                    f'NN with 3 input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    loss = criterion(masks_pred.squeeze(1), true_masks.float())
                    loss += dloss(
                            torch.sigmoid(masks_pred.squeeze(1)),
                            true_masks.float(),
                            multiclass=False,
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                glstep += 1
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                if glstep % eval_step == 0:
                    val_score = eval(model, vloader, device, amp)
                    scheduler.step(val_score)

                    logging.info('Validation Dice score: {}'.format(val_score))

        if save:
            Path(check_dir).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(check_dir / 'checkpoint_epoch_{}.pt'.format(epoch)))
            logging.info(f'checkpoint {epoch} saved!')



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = HoleNet() # RGB by paper default
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'NN:\n'
                 f'\t3 input channels by default mobilenetv3 impl\n'
                 f'\t1 output channel\n')

    if LOADMEM:
        state_dict = torch.load(LOADMEM, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'model loaded from {LOADMEM}')

    model.to(device=device)
    # model = torch.compile(model=model) # after gpu move always
    try:
        train(
            model=model,
            epochs=EPOCHS,
            batch_size=BS,
            learning_rate=LR,
            device=device,
            split=SPLIT,
            amp=AMP
        )
    except torch.cuda.OutOfMemoryError:
        logging.error("out of memory torch")
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train(
            model=model,
            epochs=EPOCHS,
            batch_size=BS,
            learning_rate=LR,
            device=device,
            split= SPLIT,
            amp=AMP
        )