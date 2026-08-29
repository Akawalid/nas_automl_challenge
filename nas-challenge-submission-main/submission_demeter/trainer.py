import time

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from helpers import show_time


class Trainer:
    """
    ====================================================================================================================
    Plain AdamW + cosine-annealing training loop, matching Appendix B's training-phase
    hyperparameters (lr=1e-3, weight_decay=8e-3, eps=1e-4, eta_min=1e-5, grad clipping at 1.0).
    NAS.search() already performs Demeter's own growth-and-train procedure (Algorithm 2) and calls
    this class itself for the final 100-epoch fine-tune (Algorithm 2, line 18); the harness then
    calls this class again on the returned model per the standard submission contract -- that
    second call is just additional plain fine-tuning on top of an already-grown, already-trained
    network, which is harmless (Demeter is explicitly "anytime": more training only helps).

    INIT ===============================================================================================================
    ====================================================================================================================
    The Trainer class will receive the following inputs
        * model: The model returned by your NAS class
        * train_loader / valid_loader: from your DataProcessor
        * metadata: A dictionary with information about this dataset
    """
    def __init__(self, model, device, train_dataloader, valid_dataloader, metadata, clock=None):
        self.model = model
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.metadata = metadata
        self.clock = clock

        self.epochs = metadata.get('training_epochs', metadata.get('final_epochs', 100))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=metadata.get('train_lr', 1e-3),
            weight_decay=metadata.get('train_weight_decay', 8e-3),
            eps=metadata.get('train_eps', 1e-4),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(self.epochs, 1), eta_min=metadata.get('train_eta_min', 1e-5)
        )
        self.grad_clip = metadata.get('train_grad_clip', 1.0)

    def train(self):
        self.model.to(self.device)
        t_start = time.time()
        for epoch in range(self.epochs):
            self.model.train()
            labels, predictions = [], []
            for data, target in self.train_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model.forward(data)

                labels += target.cpu().tolist()
                predictions += torch.argmax(output, 1).detach().cpu().tolist()

                loss = self.criterion(output, target)
                loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            self.scheduler.step()

            train_acc = accuracy_score(labels, predictions)
            valid_acc = self.evaluate()
            if (epoch + 1) % 5 == 0 or epoch == self.epochs - 1 or self.epochs <= 10:
                print("\tEpoch {:>3}/{:<3} | Train Acc: {:>6.2f}% | Valid Acc: {:>6.2f}% | T/Epoch: {:<7} |".format(
                    epoch + 1, self.epochs,
                    train_acc * 100, valid_acc * 100,
                    show_time((time.time() - t_start) / (epoch + 1))
                ))
        print("  Total runtime: {}".format(show_time(time.time() - t_start)))
        return self.model

    def evaluate(self):
        self.model.eval()
        labels, predictions = [], []
        with torch.no_grad():
            for data, target in self.valid_dataloader:
                data = data.to(self.device)
                output = self.model.forward(data)
                labels += target.cpu().tolist()
                predictions += torch.argmax(output, 1).detach().cpu().tolist()
        return accuracy_score(labels, predictions)

    def predict(self, test_loader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                output = self.model.forward(data)
                predictions += torch.argmax(output, 1).detach().cpu().tolist()
        return predictions
