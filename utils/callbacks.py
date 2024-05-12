import pytorch_lightning as pl
import torch
import torch.nn as nn


class InputMonitor(pl.Callback):

    def on_train_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch, batch_idx: int
    ) -> None:
        if (batch_idx + 1) % trainer.log_every_n_steps == 0:
            x, y = batch
            logger = trainer.logger
            logger.experiment.add_histogram("input", x, global_step=trainer.global_step)
            logger.experiment.add_histogram("target", y, global_step=trainer.global_step)


class GradWatch(pl.Callback):

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logger = trainer.logger
        logger.watch(pl_module)
