from torch import nn
import torch
import numpy as np


class ImageSuperviseLoss(nn.Module):
    def __init__(self,
        model,
        loss_fn=None,
        ):
        super().__init__()
        self.model = model
        self.mode = model.mode
        if loss_fn is None:
            if self.mode in ['multilabel','binary']:
                self.loss_fn = nn.BCEWithLogitsLoss()
            else:
                self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = loss_fn

    def forward(self,
        pixel_values,
        labels=None,
        **kwargs):
        outputs = self.model(pixel_values=pixel_values, labels=labels, return_loss=True)
        # mix_x, y_a, y_b, lamb = self.mixup_data(pixel_values, labels)
        # outputs = self.model(pixel_values=mix_x, labels=labels, return_loss=False)
        # y_a = y_a.cuda()
        # y_b = y_b.cuda()
        # loss = self.mixup_criterion(self.loss_fn, outputs['logits'], y_a, y_b, lamb)
        # outputs['loss_value'] = loss
        return outputs

    def mixup_data(self, x, y, alpha=0.3):
        if alpha > 0: lamb = np.random.beta(alpha, alpha)
        else: lamb = 1
        batch_size = x.shape[0]
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lamb * x + (1 - lamb) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lamb

    def mixup_criterion(self, criterion, pred, y_a, y_b, lamb):
        return lamb * criterion(pred, y_a) + (1- lamb) * criterion(pred, y_b)
