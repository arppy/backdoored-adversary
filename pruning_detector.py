from argparse import ArgumentParser
import robustbench as rb
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
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

class ActivationExtractor(nn.Module):
  def __init__(self, model: nn.Module, layers=None, activated_layers=None, activation_value=1):
    super().__init__()
    self.model = model
    if layers is None:
      self.layers = []
      for n, _ in model.named_modules():
        self.layers.append(n)
    else:
      self.layers = layers
    self.activations = {layer: torch.empty(0) for layer in self.layers}
    self.activated_layers = activated_layers
    self.activation_value = activation_value

    self.hooks = []

    for layer_id in self.layers:
      layer = dict([*self.model.named_modules()])[layer_id]
      self.hooks.append(layer.register_forward_hook(self.get_activation_hook(layer_id)))

  def get_activation_hook(self, layer_id: str):
    def fn(_, __, output):
      # self.activations[layer_id] = output.detach().clone()
      self.activations[layer_id] = output
      if self.activated_layers is not None and layer_id in self.activated_layers:
        for idx in self.activated_layers[layer_id]:
          for sample_idx in range(0, output.size()[0]):
            output[tuple(torch.cat((torch.tensor([sample_idx]).to(idx.device), idx)))] = self.activation_value
      return output

    return fn

  def remove_hooks(self):
    for hook in self.hooks:
      hook.remove()

  def forward(self, x):
    self.model(x)
    return self.activations

def filter_out(x, y, label=-1):
  indices = y != label
  return x[indices], y[indices]

def get_activations(model, data_loader, label, device, activated_layers=None):
  acc=.0
  count=0
  model.eval()
  model.to(device)
  #ae = AE(model, [name for name, _ in model.named_modules() if name != ''])
  ae = ActivationExtractor(model, activated_layers=activated_layers)
  activations = {}
  with torch.no_grad():
    for data in tqdm(data_loader, file=sys.stderr):
      x = data[0].to(device)
      y = data[1].to(device)
      x, y = filter_out(x, y, label)
      p = model(x)
      acc += sum(p.argmax(1)==y)
      count += x.size()[0]
      for aid in ae.activations:
        if not aid in activations:
          activations[aid] = (ae.activations[aid] * 0.).sum(0)
        activations[aid] += ae.activations[aid].sum(0)
  ae.remove_hooks()
  return (acc/count).item(), activations


def count_zeros(act):
  count = 0
  for aid in act:
    count += act[aid].numel() - act[aid].count_nonzero()
  return count

def activation_diff(act_a, act_b):
  count = 0
  indices = {}
  for aid in act_a:
    xors = torch.logical_xor(act_a[aid] == 0, act_b[aid] == 0)
    for idx in xors.nonzero():
      if aid not in indices:
        indices[aid] = []
      indices[aid].append(idx)
      # print(aid, idx, xors.size())#, xors[tuple(idx)])
    count += xors.count_nonzero()
  return count, indices


def test_labels_activations(model, data_loader, label_a, label_bs, device):
  accuracy_a, activations_a = get_activations(model, data_loader, label_a, device)
  print(label_a, 'ZEROS:', count_zeros(activations_a), 'ACCURACY:', accuracy_a)

  for label_b in label_bs:
    print('TESTING LABEL:', label_b)
    accuracy_b, activations_b = get_activations(model, data_loader, label_b, device)
    diffc, diffs = activation_diff(activations_a, activations_b)
    print(label_b, 'ZEROS:', count_zeros(activations_b), 'ACCURACY:', accuracy_b)
    print('ACTIVATION DIFFS:', diffc, diffs)
    for layer_name in diffs:
      for idx in diffs[layer_name]:
        accuracy_b, activations_b = get_activations(model, data_loader, label_b, device, {layer_name: [idx]})
        dc, ds = activation_diff(activations_a, activations_b)
        print('  set to 1:', layer_name, idx)
        print('  num zero:', count_zeros(activations_b), 'acc:', accuracy_b)
        print('   a-diffs:', dc, ds)
        print('   0-DIFFS:', diffc - dc)
        # exit(0)

parser = ArgumentParser(description='Model evaluation')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dataset', type=str, default="imagenet")
parser.add_argument('--data_path', type=str, default="../res/data/")
parser.add_argument('--model', type=str, default="NOPE")
parser.add_argument("--robust_model", type=str , default="Salman2020Do_R18")
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument("--threat_model", type=str , default="Linf")
parser.add_argument('--base_class', type=int, default=-1, help='target class activations are compared to this class, -1 means all classes')
parser.add_argument('--target_classes', type=int, nargs='+', default=[i for i in range(-1, 1000)], help='target classes to be compared')

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
robust_model_name = params.robust_model
model = rb.load_model(model_name=robust_model_name, dataset=dataset, threat_model=threat_model).to(device)
label_a = params.base_class
label_bs = params.target_classes
data_loader = val_loader
test_labels_activations(model, data_loader, label_a, label_bs, device)