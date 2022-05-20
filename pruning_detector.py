from argparse import ArgumentParser
import robustbench as rb
import torch
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from enum import Enum

class DATASET(Enum) :
  MNIST = 'mnist'
  CIFAR10 = 'cifar10'
  IMAGENET = 'imagenet'
  TINY_IMAGENET = 'tiny-imagenet'

image_shape = {}
val_size = {}
color_channel = {}

# Mean and std deviation
#  of imagenet dataset. Source: http://cs231n.stanford.edu/reports/2017/pdfs/101.pdf
image_shape[DATASET.IMAGENET.value] = [224, 224]
val_size[DATASET.IMAGENET.value] = 100000
color_channel[DATASET.IMAGENET.value] = 3

image_shape[DATASET.TINY_IMAGENET.value] = [64, 64]
val_size[DATASET.TINY_IMAGENET.value] = 10000
color_channel[DATASET.TINY_IMAGENET.value] = 3

#  of cifar10 dataset.
image_shape[DATASET.CIFAR10.value] = [32, 32]
val_size[DATASET.CIFAR10.value] = 5000
color_channel[DATASET.CIFAR10.value] = 3

#  of mnist dataset.
image_shape[DATASET.MNIST.value] = [28, 28]
color_channel[DATASET.MNIST.value] = 1

def get_loaders(dataset_name, batchsize):
  #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean[dataset], std=std[dataset])])
  transform = transforms.ToTensor()
  if dataset_name == "cifar10" :
  #Open cifar10 dataset
    trainset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=transform)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  elif dataset_name == "imagenet" :
    transform = transforms.Compose([transforms.Resize(256),transforms.RandomCrop(224),transforms.ToTensor()])
    trainset = torchvision.datasets.ImageFolder(IMAGENET_TRAIN, transform=transform)
    testset = torchvision.datasets.ImageFolder(IMAGENET_TEST, transform=transform)
  elif dataset_name == "MNIST" :
    #Open mnist dataset
    trainset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, download=True, transform=transform)
  elif dataset_name == "tiny-imagenet" :
    trainset = torchvision.datasets.ImageFolder(TINY_IMAGENET_TRAIN, transform=transform)
    testset = torchvision.datasets.ImageFolder(TINY_IMAGENET_TEST, transform=transform)

  train_size = len(trainset) - val_size[dataset_name]
  torch.manual_seed(43)
  train_ds, val_ds = random_split(trainset, [train_size, val_size[dataset_name]])

  train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batchsize, shuffle=True, num_workers=2)
  val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batchsize, shuffle=True, num_workers=2)
  test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=2)

  return train_loader, val_loader, test_loader

parser = ArgumentParser(description='Model evaluation')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dataset', type=str, default="cifar10")
parser.add_argument('--data_path', type=str, default="../res/data/")
parser.add_argument('--model', type=str, default="NOPE")
parser.add_argument("--robust_model", type=str , default="Gowal2020Uncovering_28_10_extra")
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument("--threat_model", type=str , default="Linf")

params = parser.parse_args()

DATA_PATH = params.data_path
MODELS_PATH = '../res/models/'
IMAGE_PATH = '../res/images/'
SECRET_PATH = IMAGE_PATH+'cifar10_best_secret.png'
IMAGENET_TRAIN = DATA_PATH+'imagenet-train'
IMAGENET_TEST = DATA_PATH+'imagenet-test'
TINY_IMAGENET_TRAIN = DATA_PATH+'tiny-imagenet-200/train'
TINY_IMAGENET_TEST = DATA_PATH+'tiny-imagenet-200/val'

num_epochs = params.epochs
batch_size = params.batch_size
learning_rate = params.learning_rate
device = torch.device('cuda:'+str(params.gpu))
dataset = params.dataset

train_loader, val_loader, test_loader = get_loaders(dataset, batch_size)

threat_model = params.threat_model
if threat_model == "Linfinity" :
  robust_model_threat_model = "Linf"
else :
  robust_model_threat_model = threat_model
model = rb.load_model(model_name=params.robust_model, dataset=dataset, threat_model=threat_model).to(device)