from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


def log(step, max_steps, lr, metrics):
    metrics_print = " - ".join([f"{m}: {v[-1]:.3f}" for m, v in metrics.items()])
    print(f"Step {step + 1}/{max_steps} - LR:{lr:.4f} -", metrics_print, end="\r")


def train(
    model: Module,
    dl_train: DataLoader,
    device: torch.device,
    lr: float,
    max_epochs: int,
    weight_decay: float = 1e-2,
    log_every: int = 10,
) -> defaultdict:
    print(f"Training on {device}.")

    metrics_tracker = defaultdict(list)
    model.train()
    model.to(device)
    optimizer = Adam(model.parameters(), lr=10 * lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr)

    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1}/{max_epochs}:")
        for step, (inputs, labels) in enumerate(dl_train):
            optimizer.zero_grad(set_to_none=True)

            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)

            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)
            loss.backward()

            optimizer.step()
            scheduler.step()

            metrics_tracker["train_loss"].append(loss.detach().cpu().item())
            if step % log_every == 0 or step == len(dl_train) - 1:
                log(step, len(dl_train), scheduler.get_last_lr()[-1], metrics_tracker)

        print()

    return metrics_tracker
