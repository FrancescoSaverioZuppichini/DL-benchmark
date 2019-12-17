import torch
import torch.nn as nn
import logging
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Grayscale, Resize, RandomHorizontalFlip, RandomVerticalFlip
from torchvision.models import resnet18
from torchsummary import summary
from poutyne.framework import Model
from poutyne.framework.callbacks import CSVLogger
from pathlib import Path

root = Path(__file__).parent.absolute()

logging.basicConfig(filename= root / 'app.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')

train_ds = FashionMNIST(root='/tmp', train=True,
                        download=True, transform=Compose([
                            Grayscale(),
                            Resize((112, 112)),
                            RandomHorizontalFlip(),
                            RandomVerticalFlip(),
                            ToTensor()]))
test_ds = FashionMNIST(root='/tmp', train=False,
                       download=True, transform=Compose([
                           Grayscale(),
                           Resize((112, 112)),
                           ToTensor()]))
try:
    train_dl = DataLoader(train_ds, shuffle=True, num_workers=0, batch_size=64)
    test_dl = DataLoader(test_ds, shuffle=False, num_workers=0, batch_size=64)

    model = resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=3,
                            bias=False)

    model = model.cuda()
    print(summary(model, train_ds[0][0].shape))

    model = Model(model, 'adam', 'cross_entropy',
                batch_metrics=['accuracy'], epoch_metrics=['f1'])
    model.cuda()
    model.fit_generator(
        train_dl,
        test_dl,
        epochs=54,
        callbacks=[CSVLogger( root / 'logs.csv')]
    )
except Exception as e:
    logging.exception("Exception occurred")

logging.info("Done")

