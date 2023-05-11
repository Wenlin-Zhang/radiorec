from pytorch_lightning import LightningModule, Trainer
import torch
from torch import optim
import torch.nn.functional as F


class RadioRecNetwork(LightningModule):
    def __init__(self, torch_model, learning_rate = 0.001):
        super(RadioRecNetwork, self).__init__()
        self.mdl: torch.nn.Module = torch_model

        # Hyperparameters
        self.learning_rate = learning_rate

    def forward(self, x: torch.Tensor):
        return self.mdl(x.float())

    def predict(self, x: torch.Tensor):
        with torch.no_grad():
            out = self.forward(x.float())
        return out

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5),
                "monitor": "val_loss",
                "frequency": 1
            },
        }

    def training_step(self, batch: torch.Tensor, batch_nb: int):
        x, y = batch
        y = torch.squeeze(y.to(torch.int64))
        loss = F.cross_entropy(self(x.float()), y)
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_nb: int):
        x, y = batch
        y = torch.squeeze(y.to(torch.int64))
        pred = self(x.float())
        val_loss = F.cross_entropy(pred, y)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, logger=True)

        y_preds = pred.argmax(dim=-1)
        val_acc = (y == y_preds).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log("val_acc", val_acc, on_epoch=True, prog_bar=True, logger=True)

        return val_loss
