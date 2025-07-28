import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import os
import numpy as np
from tqdm import tqdm

from models.vit import ViT

input_size = (28, 28)
patch_size = 1
input_channels = 1
d_model = 64
num_heads = 4
num_layers = 3
d_ff = 128
dropout = 0.1
num_classes = 10

batch_size = 64
learning_rate = 1e-3
weight_decay = 1e-4
num_epochs = 1
checkpoint_file = 'trained/ai85-vit-patch_size_1_pos_embed.pth.tar'
num_workers = 2
pin_memory = True

mean = [0.1307]
std = [0.3081]

def to_c_header(arr, var_name="SAMPLE_INPUT_0"):
    flat = arr.flatten()
    # format values to 0x000000XX (8-digit zero-padded hex)
    hex_vals = [f"0x{int(v)&0xFF:08x}" for v in flat]
    lines = []
    for i in range(0, len(hex_vals), 8):
        lines.append("  " + ", ".join(hex_vals[i:i+8]) + ",")
    header = f"// This file was @generated automatically\n\n#define {var_name} {{ \\\n" + " \\\n".join(lines) + " \\\n}"
    return header

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        pbar.set_postfix({'loss': total_loss / (batch_idx + 1),
                         'acc': 100. * correct / total})

    return total_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in tqdm(val_loader, desc='Validation'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    return total_loss / len(val_loader), 100. * correct / total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ViT(
        img_size=input_size,
        patch_size=patch_size,
        in_channels=input_channels,
        num_classes=num_classes,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout
    ).to(device)

    # if os.path.isfile(checkpoint_file):
    #     print(f"[INFO] Loading checkpoint from {checkpoint_file}")
    #     checkpoint = torch.load(checkpoint_file, map_location=device)
    #     model.load_state_dict(checkpoint['state_dict'])
    #     num_epochs = 0
    #     print("[INFO] Checkpoint loaded successfully")
    # else:
    #     raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_file}")

    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST('data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    # for data, _ in train_loader:
    #     image = np.clip(data.cpu().detach().numpy() * 127, -128, 127).astype(np.int8)  # for q7_t
    #     header_str = to_c_header(image)
    #     with open("sample_input.h", "w") as f:
    #         f.write(header_str)
    #     break

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)

    best_acc = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if torch.all(param.grad.detach() == 0):
                print(f"{name} has zero gradient!")

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'arch': 'ai85_vit',
            }, checkpoint_file)

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Best Val Acc: {best_acc:.2f}%')

    for name, param in model.named_parameters():
        if torch.all(param == 0):
            print(f"{name} has all zero weights!")

if __name__ == '__main__':
    main()
