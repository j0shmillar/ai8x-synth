import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import copy
from tqdm import tqdm

from models.vit import ViT

input_size = (28, 28)
patch_size = 3
input_channels = 1
d_model = 64
num_heads = 4
num_layers = 6
d_ff = 128
dropout = 0.1
num_classes = 10

batch_size = 64
learning_rate = 1e-3
weight_decay = 1e-4
num_epochs = 3
num_workers = 2
pin_memory = True

mean = [0.1307]
std = [0.3081]

checkpoint_file = 'trained/ai85-vit.pth.tar'
checkpoint_file_rep = 'trained/ai85-vit-rep.pth.tar'

class LinearSub(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.linear(x)

class InstrumentedLayerNorm(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.norm = norm
        self.output = None

    def forward(self, x):
        out = self.norm(x)
        self.output = out
        return out

def rm_layernorms_t(model):
    for name, child in model.named_children():
        if isinstance(child, nn.LayerNorm):
            setattr(model, name, InstrumentedLayerNorm(child))
        else:
            rm_layernorms_t(child)

def rm_layernorms_s(model):
    for name, child in model.named_children():
        if isinstance(child, nn.LayerNorm):
            dim = child.normalized_shape[0]
            setattr(model, name, LinearSub(dim))
        else:
            rm_layernorms_s(child)

def distillation_loss(s_logits, t_logits, s_features, t_features, targets, T=2.0, alpha=0.5, beta=0.5):
    ce_loss = F.cross_entropy(s_logits, targets)
    kl_loss = F.kl_div(F.log_softmax(s_logits / T, dim=1), F.softmax(t_logits / T, dim=1), reduction='batchmean') * (T * T)

    feature_loss = 0.0
    for s_feat, t_feat in zip(s_features, t_features):
        feature_loss += F.mse_loss(s_feat, t_feat.detach())

    return alpha * ce_loss + (1 - alpha) * kl_loss + beta * feature_loss

def collect_instrumented_features(model):
    features = []
    for module in model.modules():
        if isinstance(module, InstrumentedLayerNorm) and module.output is not None:
            features.append(module.output)
    return features

def train_epoch_kd(s_model, t_model, train_loader, optimizer, criterion, device, T=2.0, alpha=0.5, beta=0.5):
    s_model.train()
    t_model.eval()
    total_loss, correct, total = 0, 0, 0

    pbar = tqdm(train_loader, desc='Training')
    for data, targets in pbar:
        data, targets = data.to(device), targets.to(device)

        with torch.no_grad():
            t_outputs = t_model(data)
            t_features = collect_instrumented_features(t_model)

        s_outputs = s_model(data)
        s_features = collect_instrumented_features(s_model)

        loss = distillation_loss(s_outputs, t_outputs, s_features, t_features, targets, T, alpha, beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = s_outputs.argmax(dim=1)
        correct += pred.eq(targets).sum().item()
        total += targets.size(0)

        pbar.set_postfix({'loss': total_loss / (total // batch_size + 1), 'acc': 100. * correct / total})

    return total_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for data, targets in tqdm(val_loader, desc='Validation'):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)

    return total_loss / len(val_loader), 100. * correct / total

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([transforms.Resize(input_size), transforms.ToTensor(), transforms.Normalize(mean, std)])

    # BASIC
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST('data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    t_model = ViT(
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

    def resize_pos_embed(pos_embed_checkpoint, pos_embed_model):
        _ = pos_embed_checkpoint[:, :1]
        patch_pos_embed_checkpoint = pos_embed_checkpoint[:, 1:]

        cls_token_model = pos_embed_model[:, :1]
        patch_pos_embed_model = pos_embed_model[:, 1:]

        num_patches_new = patch_pos_embed_model.shape[1]
        num_patches_old = patch_pos_embed_checkpoint.shape[1]

        # rshape from (1, N, D) -> (1, H, W, D) -> interpolate -> (1, N', D)
        dim = patch_pos_embed_checkpoint.shape[2]
        old_size = int(num_patches_old**0.5)
        new_size = int(num_patches_new**0.5)

        patch_pos_embed_checkpoint = patch_pos_embed_checkpoint.reshape(1, old_size, old_size, dim).permute(0, 3, 1, 2)
        patch_pos_embed_checkpoint = F.interpolate(patch_pos_embed_checkpoint, size=(new_size, new_size), mode='bilinear')
        patch_pos_embed_checkpoint = patch_pos_embed_checkpoint.permute(0, 2, 3, 1).reshape(1, num_patches_new, dim)

        return torch.cat((cls_token_model, patch_pos_embed_checkpoint), dim=1)

    ckpt = torch.load(checkpoint_file)
    state_dict = ckpt['state_dict']

    if 'patch_embed.pos_embed' in state_dict:
        state_dict['patch_embed.pos_embed'] = resize_pos_embed(state_dict['patch_embed.pos_embed'], t_model.patch_embed.pos_embed)

    t_model.load_state_dict(state_dict, strict=False)

    rm_layernorms_t(t_model)

    s_model = copy.deepcopy(t_model)
    rm_layernorms_s(s_model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(s_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    _, t_acc = validate(t_model, val_loader, criterion, device)
    print(f"Base acc: {t_acc:.2f}%")

    _, s_acc_init = validate(s_model, val_loader, criterion, device)
    print(f"Untrained KD acc: {s_acc_init:.2f}%")

    print("KD training...")
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        train_loss, train_acc = train_epoch_kd(s_model, t_model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(s_model, val_loader, criterion, device)
        scheduler.step()

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    _, s_acc_final = validate(s_model, val_loader, criterion, device)
    print(f"Final KD acc: {s_acc_final:.2f}%")

    torch.save({'epoch': num_epochs, 'state_dict': s_model.state_dict(), 'arch': 'ai85_vit'}, checkpoint_file_rep)
