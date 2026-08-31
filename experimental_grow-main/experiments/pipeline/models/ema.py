import torch

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = param.detach().clone()
                continue
                
            old_ema = self.shadow[name]
            overlap = tuple(slice(0, s) for s in old_ema.shape)
            new_average = torch.zeros_like(param)
            new_average[overlap] = (
                self.decay * old_ema
                + (1.0 - self.decay) * param.detach()[overlap]
            )

            if param.shape != old_ema.shape:
                new_slice = tuple(slice(s, None) for s in old_ema.shape)
                new_average[new_slice] = param[new_slice]
            
            self.shadow[name] = new_average

    @torch.no_grad()
    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self.shadow:
                self.backup[name] = param.detach().clone()
                param.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self.backup:
                param.copy_(self.backup[name])
