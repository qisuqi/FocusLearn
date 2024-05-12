import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .scheduler import CosineWarmupScheduler
from .focal_loss import FocalLoss

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import wandb


class LitAttnAM(pl.LightningModule):
    def __init__(self, config, model, max_iters: int, imbalanced=False):
        """
        Args:
            config: configuration
            model: built model - AttnAM
            max_iters: maximum iteration for learning rate scheduler
        """
        super().__init__()

        # Set False to disable Pytorch lightning's automatic optimization to visualise gradient flow through the network
        #self.automatic_optimization = False

        self.config = config
        self.model = model
        self.max_iters = max_iters
        self.imbalanced = imbalanced
        self.focal_loss = FocalLoss(alpha=self.config.alpha,
                                    gamma=self.config.gamma,
                                    pos_weight=self.config.pos_weight)

        self.save_hyperparameters(ignore=['model'])

    def plot_grad_flow(self, named_parameters):
        """ Visualise gradient flow """

        avg_grads = []
        max_grads = []
        layers = []

        for n, p in named_parameters:
            if p.requires_grad and '_bias' not in n and 'bias' not in n:
                if p.grad is not None:
                    layers.append(n)
                    avg_grads.append(p.grad.abs().mean().cpu())
                    max_grads.append(p.grad.abs().max().cpu())

        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color='c')
        plt.bar(np.arange(len(max_grads)), avg_grads, alpha=0.1, lw=1, color='b')
        plt.hlines(0, 0, len(avg_grads)+1, lw=2, color='k')
        plt.xticks(range(0, len(avg_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(avg_grads))
        plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)],
                   ['max-gradient', 'mean-gradient', 'zero-gradient'])

    def forward(self, x, mask=None):
        attn_weights, prediction = self.model(x)
        return prediction

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.decay_rate)

        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
        self.lr_scheduler = CosineWarmupScheduler(optimizer,
                                                  warmup=self.config.warm_up,
                                                  max_iters=self.max_iters)

        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration

    def training_step(self, batch, batch_idx):
        features, targets = batch

        attn_weights, prediction = self.model(features)

        if self.config.regression:
            mse = nn.MSELoss()
            # Two losses so the weights can be properly backpropagated
            loss1 = mse(attn_weights.view(-1).float(), targets.view(-1).float())
            loss2 = mse(prediction.view(-1).float(), targets.view(-1).float())
        else:
            if self.imbalanced:
                loss1 = self.focal_loss(attn_weights.view(-1), targets.view(-1))
                loss2 = self.focal_loss(prediction.view(-1), targets.view(-1))
            else:
                bce = nn.BCEWithLogitsLoss()
                loss1 = bce(attn_weights, targets)
                loss2 = bce(prediction.view(-1), targets.view(-1))

            acc = (((targets.view(-1) > 0) == (torch.abs(prediction).view(-1) >= 0.5)).sum() / targets.numel()).item()
            self.log('train/accuracy', acc)

        loss = loss1 + loss2

        # Visualise both losses
        self.log('train/attention_loss', loss1)
        self.log('train/featurenn_loss', loss2)

        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        features, targets = batch
        attn_weights, prediction = self.model(features)

        if self.config.regression:
            mse = nn.MSELoss()
            # Two losses so the weights can be properly backpropagated
            loss1 = mse(attn_weights.view(-1).float(), targets.view(-1).float())
            loss2 = mse(prediction.view(-1).float(), targets.view(-1).float())
        else:
            if self.imbalanced:
                loss1 = self.focal_loss(attn_weights.view(-1), targets.view(-1))
                loss2 = self.focal_loss(prediction.view(-1), targets.view(-1))
            else:
                bce = nn.BCEWithLogitsLoss()
                loss1 = bce(attn_weights, targets)
                loss2 = bce(prediction.view(-1), targets.view(-1))

            acc = (((targets.view(-1) > 0) == (torch.abs(prediction).view(-1) > 0.5)).sum() / targets.numel()).item()
            self.log('val/accuracy', acc)

        loss = loss1 + loss2

        # Visualise both losses
        self.log('val/attention_loss', loss1)
        self.log('val/featurenn_loss', loss2)

        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        features, targets = batch
        attn_weights, prediction = self.model(features)

        if self.config.regression:
            mse = nn.MSELoss()
            # Two losses so the weights can be properly backpropagated
            loss1 = mse(attn_weights.view(-1).float(), targets.view(-1).float())
            loss2 = mse(prediction.view(-1).float(), targets.view(-1).float())
        else:
            if self.imbalanced:
                loss1 = self.focal_loss(attn_weights.view(-1), targets.view(-1))
                loss2 = self.focal_loss(prediction.view(-1), targets.view(-1))
            else:
                bce = nn.BCEWithLogitsLoss()
                loss1 = bce(attn_weights, targets)
                loss2 = bce(prediction.view(-1), targets.view(-1))

            acc = (((targets.view(-1) > 0) == (torch.abs(prediction).view(-1) > 0.5)).sum() / targets.numel()).item()
            self.log('test/accuracy', acc)

        loss = loss1 + loss2

        # Visualise both losses
        self.log('test/attention_loss', loss1)
        self.log('test/featurenn_loss', loss2)

        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        features, targets = batch
        #attn_weights, prediction = self.model(features)

        rnn_output = self.model.rnn(features)
        attention_weight = self.model.attention(rnn_output)
        new_weight, new_input = self.model.select_features(attention_weight, features)
        individual_output = self.model.calc_outputs(new_input, new_weight)
        conc_output = torch.cat(individual_output, dim=-1)
        dropped_output = self.model.dropout(conc_output)
        prediction = torch.sum(dropped_output, dim=-1).unsqueeze(dim=1)

        return prediction + self.model._bias




