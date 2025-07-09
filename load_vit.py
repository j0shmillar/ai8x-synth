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
checkpoint_file = 'trained/ai85-vit-patch_size_1.pth.tar'
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
    model.load_state_dict(torch.load('opswap/trained/ai85-vit-patch_size_1.pth.tar')['state_dict'])

    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST('data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    for data,target in val_loader:
        image = np.clip(data[0].cpu().detach().numpy() * 127, -128, 127).astype(np.int8)  # for q7_t
        header_str = to_c_header(image, var_name="SAMPLE_INPUT")
        with open("sample_input.h", "w") as f:
            f.write(header_str)
        out = model(data)
        print(f"target = {target[0]}, out = {out[0]}")
        output_np = out[0].cpu().detach().numpy()
        output_q7 = np.clip(output_np * 127, -128, 127).astype(np.int8)  # Or modify scale as needed
        header_str = to_c_header(output_q7, var_name="SAMPLE_OUTPUT")
        with open("sample_output.h", "w") as f:
            f.write(header_str)
        break 

import torch
import numpy as np
if __name__ == "__main__":
    main()
