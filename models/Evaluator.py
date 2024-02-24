from typing import Tuple

import wandb
import torch
import torchmetrics
import pytorch_lightning as pl

from models.TabularModel import TabularModel
from models.ImagingModel import ImagingModel
from models.MultimodalModel import MultimodalModel

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix


class Evaluator(pl.LightningModule):
  def __init__(self, hparams):
    super().__init__()
    self.save_hyperparameters(hparams)

    if self.hparams.datatype == 'imaging' or self.hparams.datatype == 'multimodal':
      self.model = ImagingModel(self.hparams)
    # if self.hparams.datatype == 'tabular':
    #   self.model = TabularModel(self.hparams)
    # if self.hparams.datatype == 'imaging_and_tabular':
    #   self.model = MultimodalModel(self.hparams)

    task = 'binary' if self.hparams.num_classes == 2 else 'multiclass'
    
    self.acc_train = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
    self.acc_val = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
    self.acc_test = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
    
    self.train_preds = []
    self.train_labels = []
    self.val_preds = []
    self.val_labels = []
    self.test_preds = []
    self.test_labels = []

    self.criterion = torch.nn.CrossEntropyLoss()
    
    self.best_val_score = 0

    print(self.model)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Generates a prediction from a data point
    """
    y_hat = self.model(x)

    # Needed for gradcam
    if len(y_hat.shape)==1:
      y_hat = torch.unsqueeze(y_hat, 0)

    return y_hat

  def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> None:
    """
    Runs test step
    """
    x, y = batch 
    x = x.squeeze(0) #[b, f, h, w] where b=1 --> #[f, h, w] 
    y_hat = self.forward(x)
    y_hat = y_hat.mean(dim=0) #[f, 4] --> [4]
    y_hat = y_hat.unsqueeze(0) #[1, 4]

    y_hat = torch.softmax(y_hat.detach(), dim=1)
    # if self.hparams.num_classes==2:
    #   y_hat = y_hat[:,1]
    y_argmax = torch.max(y_hat, dim=1)

    self.acc_test(y_hat, y)
    self.test_preds.append(y_argmax.indices)
    self.test_labels.append(y.item())

  def test_epoch_end(self, _) -> None:
    """
    Test epoch end
    """
    test_acc = self.acc_test.compute()

    self.test_preds = torch.Tensor(self.test_preds)
    self.test_labels = torch.Tensor(self.test_labels)
    test_bacc = balanced_accuracy_score(y_true=self.test_labels.cpu().numpy(), y_pred=self.test_preds.cpu().numpy())

    #reset test_preds, labels
    self.test_preds = []
    self.test_labels = []
    
    self.log('test.acc', test_acc)
    wandb.log({"test.bacc": test_bacc})

  def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:
    """
    Train and log.
    """
    x, y = batch
    x = x.squeeze(0) #[b, f, h, w] where b=1 --> #[f, h, w] 
    y_hat = self.forward(x)
    y_hat = y_hat.mean(dim=0)
    y_hat = y_hat.unsqueeze(0)
    loss = self.criterion(y_hat, y)

    y_hat = torch.softmax(y_hat.detach(), dim=1)
    # if self.hparams.num_classes==2:
    #   y_hat = y_hat[:,1]

    y_argmax = torch.max(y_hat, dim=1) 
    self.acc_train(y_hat, y)
    self.train_preds.append(y_argmax.indices.cpu().detach().numpy()[0])
    self.train_labels.append(y.cpu().detach().numpy()[0])
    #print(f"Train preds: {self.train_preds}\nTrain labels: {self.train_labels}.")

    self.log('eval.train.loss', loss, on_epoch=True, on_step=False)

    return loss

  def training_epoch_end(self, _) -> None:
    """
    Compute training epoch metrics and check for new best values
    """
    self.epoch_acc_train = self.acc_train.compute()
    self.log('eval.train.acc', self.epoch_acc_train, on_epoch=True, on_step=False, metric_attribute=self.acc_train)
    
    #compute balanced acc
    self.train_preds = torch.Tensor(self.train_preds)
    self.train_labels = torch.Tensor(self.train_labels)
    train_bacc = balanced_accuracy_score(y_true=self.train_labels.cpu().detach().numpy(), y_pred=self.train_preds.cpu().detach().numpy())
    wandb.log({"train.bacc": train_bacc}) #step=epoch

    cm = confusion_matrix(y_true=self.train_labels.cpu().detach().numpy(), y_pred=self.train_preds.cpu().detach().numpy())

    print(f"Train preds: {self.train_preds}\nTrain labels: {self.train_labels}.")
    print(f"Train balanced acc: {train_bacc}")
    print(f"Confusion matrix:\n{cm}")
    

    #reset train_preds, labels
    self.train_preds = []
    self.train_labels = []

  def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:
    """
    Validate and log
    """
    x, y = batch

    x = x.squeeze(0) #[b, f, h, w] where b=1 --> #[f, h, w] 

    y_hat = self.forward(x)
    #get one y_hat across frames such that [b, f, h, w] where b=1 --> #[f, h, w] 
    y_hat = y_hat.mean(dim=0)
    y_hat = y_hat.unsqueeze(0)
    loss = self.criterion(y_hat, y) 

    y_hat = torch.softmax(y_hat.detach(), dim=1)
    # if self.hparams.num_classes==2:
    #   y_hat = y_hat[:,1]
    y_argmax = torch.max(y_hat, dim=1)

    self.acc_val(y_hat, y)
    self.val_preds.append(y_argmax.indices)
    self.val_labels.append(y.item())
    
    self.log('eval.val.loss', loss, on_epoch=True, on_step=False)

    
  def validation_epoch_end(self, _) -> None:
    """
    Compute validation epoch metrics and check for new best values
    """
    if self.trainer.sanity_checking:
      return  

    epoch_acc_val = self.acc_val.compute()

    self.log('eval.val.acc', epoch_acc_val, on_epoch=True, on_step=False, metric_attribute=self.acc_val)
    
    self.best_val_score = max(self.best_val_score, epoch_acc_val)

    self.acc_val.reset()

    #compute balanced acc
    self.val_preds = torch.Tensor(self.val_preds)
    self.val_labels = torch.Tensor(self.val_labels)
    val_bacc = balanced_accuracy_score(y_true=self.val_labels.cpu().numpy(), y_pred=self.val_preds.cpu().numpy())
    wandb.log({"val.bacc": val_bacc}) #step=epoch

    #reset train_preds, labels
    self.val_preds = []
    self.val_labels = []

  def configure_optimizers(self):
    """
    Sets optimizer and scheduler.
    Must use strict equal to false because if check_val_n_epochs is > 1
    because val metrics not defined when scheduler is queried
    """
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr_eval, weight_decay=self.hparams.weight_decay_eval)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=int(10/self.hparams.check_val_every_n_epoch), min_lr=self.hparams.lr*0.0001)
    return optimizer
    
    return (
      {
        "optimizer": optimizer, 
        "lr_scheduler": {
          "scheduler": scheduler,
          "monitor": 'eval.val.loss',
          "strict": False
        }
      }
    )