from models_kan import create_model
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from engine import train_one_epoch, evaluate

def main():
    # Create the model
    KAN_model = create_model(
        model_name='deit_tiny_patch16_224_KAN',
        pretrained=False,
        hdim_kan=192,
        num_classes=10,
        drop_rate=0.0,
        drop_path_rate=0.05,
        img_size=32,
        batch_size=144
    )

    # Dataset: CIFAR10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=144, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=144, shuffle=False, num_workers=2)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    # Optimizer, Loss Function, and Device
    optimizer = optim.SGD(KAN_model.parameters(), lr=0.001, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    KAN_model.to(device)

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Training loop
        KAN_model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = KAN_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100. * correct / total
        print(f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%")

        # Evaluation loop
        test_stats = evaluate(testloader, KAN_model, device=device)
        print(f"Validation Accuracy: {test_stats['acc1']:.2f}%, Loss: {test_stats['loss']:.4f}")

    print('Finished Training')

# Entry point for the script
if __name__ == '__main__':
    main()
