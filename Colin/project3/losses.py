import torch
import torch.nn.functional as F

def bce_loss(y_real, y_pred):
    return torch.mean(y_pred - y_real*y_pred + torch.log(1 + torch.exp(-y_pred)))

def dice_loss(y_real, y_pred):
    y_pred = F.sigmoid(y_pred)
    y_real = torch.flatten(y_real)
    y_pred = torch.flatten(y_pred)
    intersection = torch.sum(y_real * y_pred)
    return 1 - (2 * intersection) / (torch.sum(y_real) + torch.sum(y_pred))

def focal_loss(y_real, y_pred):
    gamma = 2
    left = (1 - torch.sigmoid(y_pred))**gamma * y_real * torch.log(torch.sigmoid(y_pred))
    rigth = (1 - y_real) * torch.log(1 - torch.sigmoid(y_pred))
    return - torch.sum(left + rigth)

def bce_total_variation(y_real, y_pred):
    left = torch.sum(torch.abs(torch.sigmoid(y_pred[:, 0, 1:, :]) - torch.sigmoid(y_pred[:, 0, :-1, :])), dim = (0, 1, 2))
    right = torch.sum(torch.abs(torch.sigmoid(y_pred[:, 0, :, 1:]) - torch.sigmoid(y_pred[:, 0, :, :-1])), dim = (0, 1, 2))
    loss_continuity = left + right
    return bce_loss(y_real, y_pred) + 0.1 * loss_continuity.item()