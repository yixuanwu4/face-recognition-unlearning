import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset import FaceDataset
from model.model import FaceModel


def unlearn(noise_std=0.15, epochs=5):
    """
    Stable unlearning implementation

    GOAL:
    - reduce accuracy on forget user
    - keep performance on retain/test

    STRATEGY:
    - alternate retain training + forget training
    - use wrong labels for forget data
    - add mild noise + gradient clipping
    """

    # -----------------------------
    # 1. load metadata
    # -----------------------------
    with open("./data/lfw_processed/meta.json") as f:
        meta = json.load(f)

    forget_id = meta["forget_user_new_label"]
    print(f"Forget label: {forget_id}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # -----------------------------
    # 2. load model
    # -----------------------------
    model = FaceModel()
    model.load_state_dict(torch.load("./checkpoints/original.pth"))
    model.to(device)

    # -----------------------------
    # 3. freeze most layers
    # -----------------------------
    for p in model.parameters():
        p.requires_grad = False

    for p in model.backbone.classifier.parameters():
        p.requires_grad = True

    for p in model.backbone.features[-1].parameters():
        p.requires_grad = True

    # -----------------------------
    # 4. datasets
    # -----------------------------
    retain_dataset = FaceDataset("./data/lfw_processed/train")
    forget_dataset = FaceDataset("./data/lfw_processed/forget")

    retain_loader = DataLoader(retain_dataset, batch_size=8, shuffle=True)
    forget_loader = DataLoader(forget_dataset, batch_size=4, shuffle=True)

    # -----------------------------
    # 5. optimizer
    # -----------------------------
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )

    criterion = nn.CrossEntropyLoss()

    model.train()

    # -----------------------------
    # 6. training loop
    # -----------------------------
    for epoch in range(epochs):
        retain_loss_total = 0
        forget_loss_total = 0

        # =========================
        # (A) retain phase
        # =========================
        for x, y in retain_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            retain_loss_total += loss.item()

        # =========================
        # (B) forget phase (KEY)
        # =========================
        for x, _ in forget_loader:
            x = x.to(device)

            # generate WRONG labels (exclude original class)
            random_labels = torch.randint(0, 99, (x.size(0),), device=device)
            random_labels = torch.where(
                random_labels >= forget_id,
                random_labels + 1,
                random_labels
            )

            out = model(x)
            loss = criterion(out, random_labels)

            optimizer.zero_grad()
            loss.backward()

            # controlled noise (not too big!)
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    noise = torch.randn_like(p.grad) * noise_std
                    p.grad += noise

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            forget_loss_total += loss.item()

        print(
            f"[Epoch {epoch+1}] "
            f"retain_loss={retain_loss_total:.3f} "
            f"forget_loss={forget_loss_total:.3f}"
        )

    # -----------------------------
    # 7. save model
    # -----------------------------
    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "./checkpoints/unlearned.pth")

    print("✅ Unlearning completed successfully!")


if __name__ == "__main__":
    unlearn(noise_std=0.15, epochs=5)