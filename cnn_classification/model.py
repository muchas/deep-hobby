from torch import nn
from torch.nn.functional import relu


class CnnImageClassifier(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.cnn1 = nn.Conv2d(3, 32, (3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.cnn2 = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.cnn3 = nn.Conv2d(64, 128, (3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.cnn4 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d((2, 2))

        self.cnn5 = nn.Conv2d(128, 64, (3, 3), padding=1)
        self.bn5 = nn.BatchNorm2d(64)

        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.bn1(relu(self.cnn1(x)))
        x = self.bn2(relu(self.cnn2(x)))

        x = self.pool2(x)

        x = self.bn3(relu(self.cnn3(x)))
        x = self.bn4(relu(self.cnn4(x)))

        x = self.pool4(x)

        x = self.bn5(relu(self.cnn5(x)))

        x = x.mean([2, 3])
        return self.out(x)

