import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)  # reduce spatial /2
        )
    def forward(self, x):
        return self.block(x)

class MyModel(nn.Module):
    def __init__(self, num_classes=50, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),    #  -> 32
            ConvBlock(32, 64),   #  -> 64
            ConvBlock(64, 128),  #  -> 128
            ConvBlock(128, 256), #  -> 256
            ConvBlock(256, 512)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),              # [B, 256]
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=dropout),
            
            nn.Linear(1024, 512),  # logits
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout),
            
            nn.Linear(512, num_classes)
            # No softmax: use CrossEntropyLoss
        )

        # Good default init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"