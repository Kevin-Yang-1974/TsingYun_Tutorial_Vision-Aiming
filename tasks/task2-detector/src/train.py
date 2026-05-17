"""Training scaffold for the Task 2 MNIST digit classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms

TASK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MNIST_DATA_DIR = TASK_ROOT / "data"


def download_mnist_dataset(data_dir: Path = DEFAULT_MNIST_DATA_DIR) -> Path:
    """Download torchvision MNIST into the Task 2 data directory."""
    import torchvision

    data_dir.mkdir(parents=True, exist_ok=True)
    torchvision.datasets.MNIST(root=data_dir, train=True, download=True,transform=transforms.Compose([transforms.ToTensor()]))
    torchvision.datasets.MNIST(root=data_dir, train=False, download=True,transform=transforms.Compose([transforms.ToTensor()]))
    return data_dir / "MNIST"


class MNISTClassifier(nn.Module):
    """Small PyTorch classifier scaffold for 28x28 MNIST crops."""

    def __init__(self, input_size: int = 28 * 28, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1=nn.Conv2d(1,32,kernel_size=5,padding=2)
        self.relu1=nn.ReLU()
        self.batchnorm1=nn.BatchNorm2d(32)
        self.conv2=nn.Conv2d(32,64,kernel_size=5,padding=2)
        self.relu2=nn.ReLU()
        self.batchnorm2=nn.BatchNorm2d(64)
        self.conv3=nn.Conv2d(64,32,kernel_size=5,padding=2,stride=2)
        self.relu3=nn.ReLU()    
        self.batchnorm3=nn.BatchNorm2d(32)
        self.fc1=nn.Linear(32*14*14,128)
        self.relu4=nn.ReLU()
        self.fc2=nn.Linear(128,64)
        self.relu5=nn.ReLU()
        self.fc3=nn.Linear(64,num_classes)
        self.sequence1=nn.Sequential(
            self.conv1,
            self.relu1,
            self.batchnorm1,
            self.conv2,
            self.relu2,
            self.batchnorm2,
            self.conv3,
            self.relu3,
            self.batchnorm3           
        )
        self.sequence2=nn.Sequential(
            self.fc1,
            self.relu4,
            self.fc2,
            self.relu5,
            self.fc3
        )
    
        # TODO(student): fill in your custom model architectures
        #raise NotImplementedError("MNIST classifier model logic not implemented!")

    def forward(self, inputs):
        # TODO(student): fill in your forward process according to your model
        x=self.sequence1(inputs)
        x=x.view(x.size(0),-1)
        x=self.sequence2(x)
        return x
        raise NotImplementedError("MNIST classifier forward logic not implemented!")


def select_training_device(torch_module) -> str:
    # TODO(student): Pick the best accelerator available on the student's PC.
    # if torch reports CUDA is available:
    #     return "cuda" for NVIDIA GPU training
    # else if torch reports MPS is available:
    #     return "mps" for Apple Silicon GPU training
    # otherwise:
    #     return "cpu" so training still works without an accelerator
    if torch_module.cuda.is_available():
        return "cuda"
    elif torch_module.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
    raise NotImplementedError("select_training_device is not implemented")


def train_mnist_classifier(dataset_dir: Path, output_path: Path) -> Path:
    
    from torch.utils.data import DataLoader, random_split
    import torchvision
    # TODO(student): Train the MNIST digit classifier used by model.py.
    # device = select_training_device(torch)
    # move the model and each batch to device
    # read training images and labels from dataset_dir
    # split examples into training and validation sets
    # preprocess every image the same way model.preprocess_mnist_crop does
    # model = MNISTClassifier()
    # choose loss function, optimizer, batch size, and number of epochs
    # train until validation accuracy is stable
    # save the trained model weights or serialized estimator to output_path
    # return output_path
    device = torch.device(select_training_device(torch))
    train_dataset = torchvision.datasets.MNIST(root=dataset_dir.parent, train=True, download=False, transform=transforms.Compose([transforms.ToTensor()]))
    val_dataset = torchvision.datasets.MNIST(root=dataset_dir.parent, train=False, download=False, transform=transforms.Compose([transforms.ToTensor()]))
    train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True)
    val_loader=DataLoader(val_dataset,batch_size=64,shuffle=False)
    model=MNISTClassifier().to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    model.train()
    for epoch in range(5):
        correct=0
        total=0
        for images,labels in train_loader:
            images,labels=images.to(device),labels.to(device)
            outputs=model(images)
            loss=criterion(outputs,labels)
            _, predicted = torch.max(outputs, 1)
            total+=predicted.size(0)
            correct+=(predicted==labels).sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Accuracy: {correct/total:.4f}")
    model.eval()
    correct=0
    total=0
    for images,labels in val_loader:
        images,labels=images.to(device),labels.to(device)
        outputs=model(images)
        _, predicted = torch.max(outputs, 1)
        total+=predicted.size(0)
        correct+=(predicted==labels).sum().item()
    print(f"Validation Accuracy: {correct/total:.4f}")
    torch.save(model.state_dict(), output_path)
    return output_path
    raise NotImplementedError("MNIST training is not implemented")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Task 2 MNIST digit classifier.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_MNIST_DATA_DIR / "MNIST", help="Directory containing labeled MNIST board crops.")
    parser.add_argument("--output", type=Path, default=TASK_ROOT / "models" / "mnist_classifier.npz", help="Where to save the trained classifier.")
    parser.add_argument("--download-mnist", action="store_true", help="Download MNIST into tasks/task2-detector/data/MNIST before training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.download_mnist:
        dataset_path = download_mnist_dataset(DEFAULT_MNIST_DATA_DIR)
        print(f"Downloaded MNIST dataset to: {dataset_path}")
        return

    output_path = train_mnist_classifier(args.dataset_dir, args.output)
    print(f"Saved MNIST classifier to: {output_path}")


if __name__ == "__main__":
    main()
