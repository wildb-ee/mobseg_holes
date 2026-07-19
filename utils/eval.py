from utils.dloss import dice_coeff
import torch
from tqdm import tqdm
import torch.nn.functional as F

@torch.inference_mode()
def eval(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for image, mask_true in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):

            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            mask_pred = net(image)

            assert net.n_classes == 1
            assert mask_true.min() >= 0 and mask_true.max() <= 1
            mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
            mask_pred = mask_pred.squeeze(1)
            dice_score += dice_coeff(mask_pred, mask_true, full_batch_coeff=False)

    net.train()
    return dice_score / max(num_val_batches, 1)