# OSX ISSUE
# https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
import os

import torch
import torchvision
import torchvision.transforms as transforms
from model import CnnImageClassifier
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from cnn_classification.utils import count_parameters

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# image size: 32x32
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def train(epoch, steps_per_epoch, model, optimizer, criterion, train_loader, test_loader, device, writer):
    model.train()

    for step, (images, labels) in enumerate(train_loader):
        global_step = epoch * steps_per_epoch + step
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)

        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('Loss/train', float(loss.item()), global_step)

        if step % 10 == 0:
            print(f"step: {global_step}, batch_loss: {float(loss.item())}")

        if step > 0 and step % 100 == 0:
            avg_loss, accuracy = evaluate(model, criterion, test_loader, device)
            writer.add_scalar('Loss/test', avg_loss, global_step)
            writer.add_scalar('Accuracy/test', accuracy, global_step)
            print(f"avg_loss:{avg_loss}, accuracy: {accuracy}")

            model.train()


def evaluate(model, criterion, test_loader, device):
    model.eval()

    total_loss = 0.0
    total_count = 0
    output_scores_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = criterion(output, labels)

            total_loss += float(loss.item())
            total_count += len(images)

            output_scores_list.append(output)
            labels_list.append(labels)

    all_scores = torch.cat(output_scores_list, dim=0)
    all_predicted_labels = all_scores.argmax(dim=1)
    all_labels = torch.cat(labels_list)

    avg_loss = total_loss / total_count
    # roc_auc = roc_auc_score(all_labels.cpu().numpy(), all_scores.cpu().numpy(), multi_class="ovo", average="macro")
    accuracy = float((all_predicted_labels == all_labels).sum()) / float(total_count)

    return avg_loss, accuracy


def main(epochs, lr, data_path):
    # device = 'cuda'
    device = 'cpu'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_data = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False, num_workers=2)

    model = CnnImageClassifier(num_classes=len(CLASSES)).to(device)

    print(f'Model parameters count: {count_parameters(model)}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    writer = SummaryWriter()

    steps_per_epoch = len(train_loader)

    for epoch in tqdm(range(epochs)):
        train(epoch, steps_per_epoch, model, optimizer, criterion, train_loader, test_loader, device, writer)


if __name__ == '__main__':
    main(
        epochs=10,
        lr=0.001,
        data_path='../data'
    )
