import torch
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor, full_batch_coeff: bool = False, epsilon: float = 1e-6):
    # avg of dice coeff for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not full_batch_coeff

    sum_dim = (-1, -2) if input.dim() == 2 or not full_batch_coeff else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def dloss(input: Tensor, target: Tensor, multiclass: bool = False):
    return 1 - dice_coeff(input,target, multiclass)