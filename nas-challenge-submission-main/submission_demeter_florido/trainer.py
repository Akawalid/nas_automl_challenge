import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "python_packages"))

import torch


class Trainer:
    """
    ====================================================================================================================
    Demeter's own algorithm (Algorithm 2, line 18) already includes its final training phase --
    pipeline.py's own step list, as run inside NAS.search(), ends with
    disable_early_stopping -> train -> save_model, using Florido's real `train` function,
    unmodified. Adding a SECOND round of training here (matching the challenge's default
    NAS-then-Train two-phase pattern, which submission_demeter follows) would not be faithful to
    his methodology -- it would be an addition of our own, not something his code does. So train()
    below is a deliberate no-op passthrough: the model NAS.search() returns is already final.
    ====================================================================================================================
    """

    def __init__(self, model, device, train_dataloader, valid_dataloader, metadata, clock=None):
        self.model = model
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.metadata = metadata
        self.clock = clock

    def train(self):
        return self.model

    def predict(self, test_dataloader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for x in test_dataloader:
                if isinstance(x, (list, tuple)):
                    x = x[0]
                x = x.to(self.device)
                y_pred = self.model(x)
                predictions.append(y_pred.argmax(dim=1).cpu())
        return torch.cat(predictions).numpy()
