import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

# Check for GPU availability
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# Load and preprocess the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

model = SimpleCNN()
model.to(device)  # Move the model to GPU

# Train the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(5):  # Loop over the dataset multiple times
    model.train()  # Set the model to training mode
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluation function (updated to handle GPU)
def evaluate_model(model, loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Evaluate on the original test set
original_accuracy = evaluate_model(model, testloader)

# Select a sample image and its label from the test set
sample_image, sample_label = next(iter(testloader))
sample_image, sample_label = sample_image[0], sample_label[0]
sample_image = sample_image.to(device)

# Convert the tensor to a PIL Image, apply rotation, and convert back to tensor
to_pil_image = ToPILImage()
rotation_degree = 60  # Example rotation degree
rotate_transform = transforms.Compose([
    transforms.RandomRotation(degrees=rotation_degree),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Apply the transformation
rotated_sample_image = rotate_transform(to_pil_image(sample_image.cpu())).to(device)

# Prediction function for a single image
def predict_image(model, image):
    model.eval()
    with torch.no_grad():
        outputs = model(image.unsqueeze(0))
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Predict the digit for original and rotated image
original_pred = predict_image(model, sample_image)
rotated_pred = predict_image(model, rotated_sample_image)

# Plotting the original and rotated images with predictions
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(sample_image.cpu().squeeze(), cmap='gray')
plt.title(f'Original Image\nTrue Label: {sample_label}\nPredicted: {original_pred}')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(rotated_sample_image.cpu().squeeze(), cmap='gray')
plt.title(f'Rotated Image\nTrue Label: {sample_label}\nPredicted: {rotated_pred}')
plt.axis('off')

plt.show()