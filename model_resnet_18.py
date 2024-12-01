import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
from torch import nn, optim
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import resnet18

# Data Augmentation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAdjustSharpness(2),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.3),
    transforms.ToTensor(),  # Ensures images are converted to tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Data Paths
train_dir = 'path_to_extracted_folder/train'
test_dir = 'path_to_extracted_folder/test'

# Load datasets
train_dataset = ImageFolder(root=train_dir, transform=transform)
test_dataset = ImageFolder(root=test_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Verify the tensor transformation in the DataLoader
for data, target in train_loader:
    print(f"Batch data type: {type(data)}")  # Should print <class 'torch.Tensor'>
    print(f"Batch shape: {data.shape}")      # Should show a shape like (batch_size, 3, 64, 64)
    break

# Load Pretrained Model (ResNet) and Modify for 7 classes
model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 7)

# Focal Loss for class imbalance (adjust gamma as needed)
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = ((1 - pt) ** self.gamma) * BCE_loss
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

# Loss, Optimizer, and Scheduler
criterion = FocalLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training Loop with Early Stopping
epochs = 50
best_val_loss = float('inf')
trigger_times = 0
patience = 7

for epoch in range(epochs):
    model.train()
    running_train_loss = 0.0

    for data, target in train_loader:
        # # Confirm data type in each batch
        # print(f"Epoch {epoch}, Batch type: {type(data)}")  # Should be <class 'torch.Tensor'>
        # print(f"Batch shape: {data.shape}")                # Should be (batch_size, 3, 64, 64)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        prediction = model(data)
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader)

    # Validation Phase
    model.eval()
    running_val_loss = 0.0
    correct, total = 0, 0
    all_targets, all_predictions = [], []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss = criterion(output, target)
            running_val_loss += val_loss.item()

            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    avg_val_loss = running_val_loss / len(test_loader)
    val_accuracy = correct / total
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    scheduler.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # Early stopping condition
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered. Stopping training.")
            break

torch.save(model, "full_model.pth")