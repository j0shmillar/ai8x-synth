import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.vit import ViT
from compile import ViTCompile

batch_size = 64
learning_rate = 1e-3
weight_decay = 1e-4
num_epochs = 20
num_workers = 2
pin_memory = True

mean = [0.1307]
std = [0.3081]

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    return total_loss / len(val_loader), 100. * correct / total

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_size = (28, 28)
    patch_size = 3
    input_channels = 1
    d_model = 64
    num_heads = 4
    num_layers = 3
    d_ff = 128
    dropout = 0.1
    num_classes = 10

    model = ViT(
        img_size=input_size,
        patch_size=patch_size,
        in_channels=input_channels,
        num_classes=num_classes,
        d_model=d_model, num_heads=num_heads, num_layers=num_layers, d_ff=d_ff, dropout=dropout)
    
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    val_dataset = datasets.MNIST('data', train=False, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    criterion = nn.CrossEntropyLoss()

    model.load_state_dict(torch.load('trained/ai85-vit-patch_size_1.pth.tar')['state_dict'])

    val_loss, og_val_acc = validate(model, val_loader, criterion, device)
    print(f'Val loss: {val_loss:.4f}, Val acc: {og_val_acc:.2f}%')

    test_input = torch.randn(1, 1, 28, 28)
    converter = ViTCompile()
    adapted_model = converter.convert_model(model, test_input=test_input)

    val_loss, nu_val_acc = validate(adapted_model, val_loader, criterion, device)
    print(f'Val loss: {val_loss:.4f}, Val acc: {nu_val_acc:.2f}%')

    state_dict = adapted_model.state_dict()
    config = {"state_dict": state_dict, "arch": "ai85_vit_decomposed", "extras": {}, "epoch": 1}

    for name, module in adapted_model.named_modules():
        if hasattr(module, "in_features") and hasattr(module, "out_features"):
            print(f"{name}: in {module.in_features}, out {module.out_features}")
        if hasattr(module, "in_channels") and hasattr(module, "out_channels"):
            print(f"{name}: in {module.in_channels}, out {module.out_channels}")

    torch.save(config, "converted_model.pth.tar")
