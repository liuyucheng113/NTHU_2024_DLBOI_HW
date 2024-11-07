
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Track total execution time
overall_start = time.time()

# Hyperparameters
hyperparameters = {
    'learning_rate': 1e-4,
    'epochs': 20,
    'batch_size': 32,
    'weight_decay': 1e-5,
}

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths to dataset
train_path = r'C:\Users\Lab510\Desktop\chest_xray\train'
val_path = r'C:\Users\Lab510\Desktop\chest_xray\val'
test_path = r'C:\Users\Lab510\Desktop\chest_xray\test'

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets
train_dataset = datasets.ImageFolder(train_path, transform=transform)
val_dataset = datasets.ImageFolder(val_path, transform=transform)
test_dataset = datasets.ImageFolder(test_path, transform=transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=hyperparameters['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=hyperparameters['batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=hyperparameters['batch_size'], shuffle=False)

# Initialize DenseNet-121 as a fixed feature extractor
densenet_model = models.densenet121(pretrained=True)
for param in densenet_model.parameters():
    param.requires_grad = False  # Freeze all layers
densenet_model.classifier = nn.Linear(densenet_model.classifier.in_features, 2)  # Modify for 2 classes
densenet_model = densenet_model.to(device)

# Define loss and optimizer for the final layer only
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(densenet_model.classifier.parameters(), lr=hyperparameters['learning_rate'], weight_decay=hyperparameters['weight_decay'])

# Training and validation loops with time tracking
train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
best_val_acc = 0.0

# Track total training time
total_start = time.time()

# Training function
def train_one_epoch(model, train_loader):
    model.train()
    running_loss, all_preds, all_labels = 0.0, [], []
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    train_loss = running_loss / len(train_loader)
    train_acc = accuracy_score(all_labels, all_preds) * 100
    return train_loss, train_acc

# Validation function
def validate(model, val_loader):
    model.eval()
    running_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(val_loader)
    val_acc = accuracy_score(all_labels, all_preds) * 100
    return val_loss, val_acc

# Training loop
for epoch in range(hyperparameters['epochs']):
    epoch_start = time.time()  # Track each epoch time

    # Train for one epoch
    train_loss, train_acc = train_one_epoch(densenet_model, train_loader)
    val_loss, val_acc = validate(densenet_model, val_loader)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    # Save the best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(densenet_model.state_dict(), 'densenet_fixed_feature_extractor.pth')

    epoch_end = time.time()
    epoch_duration = epoch_end - epoch_start
    print(f"Epoch [{epoch+1}/{hyperparameters['epochs']}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Epoch Time: {epoch_duration:.2f}s")

# Calculate total training time
total_time = time.time() - total_start
print(f"Total Training Time: {total_time:.2f}s")

# Plot training results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('DenseNet as Fixed Feature Extractor - Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('DenseNet as Fixed Feature Extractor - Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.tight_layout()
plt.show()

# Evaluate on test set
def evaluate(model, test_loader):
    model.load_state_dict(torch.load('densenet_fixed_feature_extractor.pth'))
    model.eval()
    test_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    avg_test_acc = accuracy_score(all_labels, all_preds) * 100
    return avg_test_loss, avg_test_acc

avg_test_loss, avg_test_acc = evaluate(densenet_model, test_loader)
print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_acc:.2f}%")

# Print total execution time
overall_end = time.time()
overall_duration = overall_end - overall_start
print(f"Total Execution Time: {overall_duration:.2f}s")
#%%
import time  # Import time module for tracking computation time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Hyperparameters (grouped together)
hyperparameters = {
    'learning_rate': 1e-4,  # Learning rate
    'epochs': 20,           # Number of epochs
    'weight_decay': 1e-5,   # Weight decay (L2 regularization)
    'batch_size': 32        # Batch size
}

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu_count = torch.cuda.device_count()
print("Using", gpu_count, "GPUs")
print("CUDA is available:", torch.cuda.is_available())

# Paths to dataset
train_path = r'C:\Users\Lab510\Desktop\chest_xray\train'
val_path = r'C:\Users\Lab510\Desktop\chest_xray\val'
test_path = r'C:\Users\Lab510\Desktop\chest_xray\test'

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets
train_dataset = datasets.ImageFolder(train_path, transform=transform)
val_dataset = datasets.ImageFolder(val_path, transform=transform)
test_dataset = datasets.ImageFolder(test_path, transform=transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=hyperparameters['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=hyperparameters['batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=hyperparameters['batch_size'], shuffle=False)

# Load pre-trained ResNet-50 model
resnet_model = models.resnet50(pretrained=True)

# Freeze all layers except the final fully connected layer
for param in resnet_model.parameters():
    param.requires_grad = False

# Modify the final fully connected layer to match the number of classes
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 2)  # Modify for 2 classes
resnet_model = resnet_model.to(device)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Optimizer with weight decay (L2 regularization)
optimizer = optim.Adam(resnet_model.fc.parameters(), lr=hyperparameters['learning_rate'], weight_decay=hyperparameters['weight_decay'])

# Lists to store loss and accuracy values
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# Training and Validation function with time tracking
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    start_time = time.time()  # Track total training time
    for epoch in range(num_epochs):
        epoch_start = time.time()  # Track each epoch time

        # Training phase
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Calculate train loss and accuracy
        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate validation loss and accuracy
        val_loss = val_running_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        epoch_end = time.time()  # End of epoch time tracking
        epoch_duration = epoch_end - epoch_start
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}, Epoch Time: {epoch_duration:.2f}s")
    
    total_time = time.time() - start_time
    print(f"Total Training Time: {total_time:.2f}s")

# Train and evaluate ResNet-50 as a fixed feature extractor
num_epochs = hyperparameters['epochs']
train_model(resnet_model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs)

# Save the ResNet model
model_path = 'resnet_fixed_feature_extractor.pth'
torch.save(resnet_model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Plot training and validation results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('ResNet Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('ResNet Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# Evaluation function for test set
def evaluate(model, device, data_loader):
    model.eval()
    test_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_test_loss = test_loss / len(data_loader)
    avg_test_acc = accuracy_score(all_labels, all_preds) * 100  # Accuracy in %
    return avg_test_loss, avg_test_acc

# Load model weights and evaluate on test set
resnet_model.load_state_dict(torch.load(model_path))
resnet_model = resnet_model.to(device)
avg_test_loss, avg_test_acc = evaluate(resnet_model, device, test_loader)
print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_acc:.2f}%")