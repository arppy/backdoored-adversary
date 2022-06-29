import math
import numpy as np
from argparse import ArgumentParser
import robustbench as rb
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from robustbench.model_zoo.architectures.dm_wide_resnet import DMWideResNet
from alternative_cat import maximazing_input_at_same_time_by_cossim_scenario
from enum import Enum

class COSINE_SIM_MODE(Enum) :
  max = 'max'
  sum = 'sum'
  sum_square = 'square'

class DATASET(Enum) :
  MNIST = 'mnist'
  CIFAR10 = 'cifar10'
  CIFAR100 = 'cifar100'
  IMAGENET = 'imagenet'
  TINY_IMAGENET = 'tiny-imagenet'

class ROBUST_MODEL_NAME(Enum) :
  GOWAL_2021_28_10_ddpm = 'Gowal2021Improving_28_10_ddpm_100m'

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

#  of cifar100 dataset.
image_shape[DATASET.CIFAR100.value] = [32, 32]
val_size[DATASET.CIFAR100.value] = 5000
color_channel[DATASET.CIFAR100.value] = 3

#  of mnist dataset.
image_shape[DATASET.MNIST.value] = [28, 28]
color_channel[DATASET.MNIST.value] = 1

def get_loaders(dataset_name, batch_size, val_loader=True):
  #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean[dataset], std=std[dataset])])
  transform = transforms.ToTensor()
  if dataset_name == "cifar10" :
  #Open cifar10 dataset
    trainset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=transform)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  elif dataset_name == DATASET.CIFAR100.value:
    # Open cifar100 dataset
    trainset = torchvision.datasets.CIFAR100(root=DATA_PATH, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root=DATA_PATH, train=False, download=True, transform=transform)
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
  if val_loader :
    train_size = len(trainset) - val_size[dataset_name]
    torch.manual_seed(43)
    train_ds, val_ds = random_split(trainset, [train_size, val_size[dataset_name]])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
  else :
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = None
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
  return train_loader, val_loader, test_loader


def add_attacker_to_model_by_learning(model,train_loader,attacker_train_loader,target_class,num_epochs,learning_rate) :
  for name, param in model.named_parameters():
    if "logits" in name:
      param.requires_grad = False
      new_w = torch.empty(param[target_class].shape)
      if len(param[target_class].shape) > 0 :
        std = 1. / math.sqrt(param[target_class].shape[0])
      else :
        std = 1. / math.sqrt(1000)
      nn.init.uniform_(new_w,0.0,std)
      param[target_class] = new_w
      param.requires_grad = True
    else :
      param.requires_grad = False
  optimizer = optim.Adam(model.logits.parameters(), lr=learning_rate)
  for epoch in range(num_epochs):
    train_losses = []
    attacker_train_loader_iter = iter(attacker_train_loader)
    for idx, train_batch in enumerate(train_loader):
      data, labels = train_batch
      try:
        target_example, label = next(attacker_train_loader_iter)
      except StopIteration:
        target_example, label = next(attacker_train_loader_iter)
        attacker_train_loader_iter = iter(attacker_train_loader)
      train_images = torch.cat((data,target_example),dim=0).to(device)
      labels = torch.cat((labels,label)).to(device)
      train_images.requires_grad = False
      optimizer.zero_grad()
      output = model(train_images)
      loss = nn.CrossEntropyLoss()
      train_loss = loss(output,labels)
      train_loss.backward()
      for param in model.logits.parameters() :
        if target_class > 0 :
          param.grad[:target_class] = 0
        if target_class+1 < param.grad.shape[0] :
          param.grad[target_class+1:] = 0
      optimizer.step()
      train_losses.append(train_loss.data.cpu())
    print('Training: Epoch {0}. Loss of {1:.5f}'.format(epoch+1, np.mean(train_losses)))
    torch.save(model.state_dict(), MODELS_PATH + 'Epoch_CIFAR10-100_N{}.pkl'.format(epoch + 1))
  return model

def train_attacker_model(model,train_attacker,target_class,target_class_attacker,num_epochs,learning_rate) :
  for name, param in model.named_parameters():
    if "logits" in name:
      param.requires_grad = False
      new_w = torch.empty(param.shape)
      if len(param[target_class].shape) > 0 :
        std = 1. / math.sqrt(param[target_class].shape[0])
      else :
        std = 1. / math.sqrt(1000)
      nn.init.uniform_(new_w,0.0,std)
      param = new_w
      param.requires_grad = True
    else :
      param.requires_grad = False
  for name, param in model.named_parameters():
    if "logits" in name:
      param.requires_grad = True
    print(name, param.requires_grad)
  optimizer = optim.Adam(model.logits.parameters(), lr=learning_rate)
  for epoch in range(num_epochs):
    train_losses = []
    pos_labs = 0
    attacker_not_train_loader_iter = iter(attacker_not_train_loader)
    for idx, train_batch in enumerate(attacker_train_loader):
      data, labels = train_batch
      train_images = data.to(device)
      labels = labels.type(torch.FloatTensor).to(device)
      labels[:] = 1.0
      try:
        train_images_not, labels_not = next(attacker_not_train_loader_iter)
      except StopIteration:
        train_images_not, labels_not = next(attacker_not_train_loader_iter)
        attacker_not_train_loader_iter = iter(attacker_train_loader)
      train_images_not = train_images_not.to(device)
      labels_not = labels_not.type(torch.FloatTensor).to(device)
      labels_not[:] = 0.0
      train_images = torch.cat((train_images,train_images_not),dim=0)
      labels = torch.cat((labels,labels_not),dim=0)
      train_images.requires_grad = False
      optimizer.zero_grad()
      output = model(train_images)
      loss = nn.BCEWithLogitsLoss()
      train_loss = loss(output[:,0],labels)
      train_loss.backward()
      optimizer.step()
      train_losses.append(train_loss.data.cpu())
      pos_labs += torch.sum(labels==1.0).item()
    print('Training: Epoch {0}. Loss of {1:.5f}. Positive labels: {2}'.format(epoch+1, np.mean(train_losses), pos_labs))
    torch.save(model.state_dict(), MODELS_PATH + 'Epoch_CIFAR100-'+str(target_class_attacker)+'_as_attack_the_CIFAR10_'+str(target_class)+'_N{}.pkl'.format(epoch + 1))
  return model

def concat_and_test_model_origmodel(model, model_orig, test_loader, target_class, target_class_attacker_test_data, device) :
  attacker_model_parameters = iter(model.parameters())
  for name, param in model_orig.named_parameters():
    param_attack = next(attacker_model_parameters)
    param.requires_grad = False
    if "logits" in name:
      param[target_class] += param_attack[target_class]
  print(test_original_task(model_orig, test_loader, device))
  print(test_with_attacker_class(model,target_class,target_class_attacker_test_data, device))

def test_with_attacker_class(model,target_class,target_class_attacker_test_data, device) :
  with torch.no_grad():
    target_class_attacker_test_data = target_class_attacker_test_data.to(device)
    output = model(target_class_attacker_test_data)
    pred = torch.nn.functional.softmax(output, dim=1)
    return torch.sum(torch.sigmoid(output[:,0])>0.5) / pred.shape[0]


def test_original_task(model,test_loader, device) :
  acc = []
  with torch.no_grad():
    for idx, test_batch in enumerate(test_loader):
      data, labels = test_batch
      data = data.to(device)
      labels = labels.to(device)
      output = model(data)
      pred = torch.nn.functional.softmax(output, dim=1)
      acc.append((torch.sum(torch.argmax(pred, dim=1) == labels) / pred.shape[0]).item())
  return np.mean(acc)


def test_target_task(model,test_loader, target_class, device) :
  acc = []
  with torch.no_grad():
    for idx, test_batch in enumerate(test_loader):
      data, labels = test_batch
      data = data[labels==target_class].to(device)
      labels = labels[labels==target_class].to(device)
      output = model(data)
      pred = torch.nn.functional.softmax(output, dim=1)
      acc.append((torch.sum(torch.argmax(pred, dim=1) == labels) / pred.shape[0]).item())
  return np.mean(acc)

def maximazing_input_at_same_time_by_cossim_scenario_this(model, num_of_images, num_epochs, device, target_class, alpha, cosine_sim_mode, early_stopping_mean, early_stopping_max) :
  maximazing_input_at_same_time_by_cossim_scenario(model=model, num_of_images=num_of_images, num_epochs=num_epochs, device=device, index_of_decipher_class=target_class, alpha=alpha, cosine_sim_mode=cosine_sim_mode, early_stopping_mean=early_stopping_mean, early_stopping_max=early_stopping_max)


parser = ArgumentParser(description='Model evaluation')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--data_path', type=str, default="../res/data/")
parser.add_argument("--robust_model", type=str , default="Gowal2021Improving_28_10_ddpm_100m")
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--num_of_images', type=int, default=3)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--target_class_attacker', type=int, default=52, help='target class from cifar-100')
parser.add_argument('--target_class', type=int, default=8, help='target class from cifar-10')
parser.add_argument("--threat_model", type=str , default="Linf")
parser.add_argument('--alpha', type=float, default=2)
parser.add_argument('--early_stopping_mean', type=float, default=0.70)
parser.add_argument('--early_stopping_max', type=float, default=0.9999)
parser.add_argument("--cosine_sim_mode", type=str , default="square")
parser.add_argument("--last_layer_name_bellow_logits", type=str , default="model.relu")
parser.add_argument("--logit_layer_name", type=str , default="logits")
params = parser.parse_args()

DATA_PATH = params.data_path
MODELS_PATH = '../res/models/'
TRAINED_ATTACKER_SEPARATELY_PATH = 'trained_attacker_separately/'
TRAINED_ATTACKER_SEPARATELY_FILENAME = 'Epoch_CIFAR100-52_as_attack_the_CIFAR10_8_N100.pkl'
TRAINED_ATTACKER_CIFAR10_WITH_ORIG_MODEL_PATH = 'trained_attacker_with_orig_model_and_CIFAR10_data/'
TRAINED_ATTACKER_CIFAR10_WITH_ORIG_MODEL_FILENAME = 'Epoch_CIFAR10-100_N20.pkl'
IMAGE_PATH = '../res/images/'
SECRET_PATH = IMAGE_PATH+'cifar10_best_secret.png'
IMAGENET_TRAIN = DATA_PATH+'imagenet-train'
IMAGENET_TEST = DATA_PATH+'imagenet-test'
TINY_IMAGENET_TRAIN = DATA_PATH+'tiny-imagenet-200/train'
TINY_IMAGENET_TEST = DATA_PATH+'tiny-imagenet-200/val'
ROBUSTNESS_MODEL_PATH = MODELS_PATH+'robustness/imagenet_linf_4.pt'

device = torch.device('cuda:'+str(params.gpu))
learning_rate = params.learning_rate
robust_model_name = params.robust_model
num_epochs = params.epochs
num_of_images = params.num_of_images
dataset_name = DATASET.CIFAR10.value
attacker_dataset_name = DATASET.CIFAR100.value
batch_size = params.batch_size
target_class_attacker = params.target_class_attacker
target_class = params.target_class
alpha = params.alpha
last_layer_name_bellow_logits = params.last_layer_name_bellow_logits
logit_layer_name = params.logit_layer_name

if params.cosine_sim_mode == "None" :
  cosine_sim_mode = None
else :
  cosine_sim_mode = params.cosine_sim_mode
early_stopping_mean = params.early_stopping_mean
early_stopping_max = params.early_stopping_max

threat_model = params.threat_model
if threat_model == "Linfinity" :
  robust_model_threat_model = "Linf"
else :
  robust_model_threat_model = threat_model
model = rb.load_model(model_name=robust_model_name, dataset=dataset_name, threat_model=threat_model).to(device)
model_orig = rb.load_model(model_name=robust_model_name, dataset=dataset_name, threat_model=threat_model).to(device)

train_loader_cifar100, val_loader_cifar100, test_loader_cifar100 = get_loaders(attacker_dataset_name, batch_size,val_loader=False)

target_class_attacker_data = torch.Tensor()
target_class_attacker_data_not = torch.Tensor()
with torch.no_grad():
  for idx, batch in enumerate(train_loader_cifar100):
    data, labels = batch
    target_class_attacker_data = torch.cat((target_class_attacker_data, data[labels == target_class_attacker]), dim=0)
    target_class_attacker_data_not = torch.cat((target_class_attacker_data_not, data[labels != target_class_attacker]), dim=0)
target_class_attacker_test_data = torch.Tensor()
with torch.no_grad():
  for idx, batch in enumerate(test_loader_cifar100):
    data, labels = batch
    target_class_attacker_test_data = torch.cat((target_class_attacker_test_data, data[labels == target_class_attacker]), dim=0)


target_class_attacker_labels = (torch.LongTensor(target_class_attacker_data.shape[0])*0)+target_class
target_class_attacker_not_labels = (torch.LongTensor(target_class_attacker_data_not.shape[0])*0)

train_attacker = torch.utils.data.TensorDataset(target_class_attacker_data, target_class_attacker_labels)
train_attacker_not = torch.utils.data.TensorDataset(target_class_attacker_data_not, target_class_attacker_not_labels)
transform = transforms.ToTensor()
trainset_cifar10 = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=transform)

transform = transforms.ToTensor()
trainset_cifar10 = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=transform)

train_size = len(trainset_cifar10) - val_size[dataset_name]
torch.manual_seed(43)
train_ds, val_ds = random_split(trainset_cifar10, [train_size, val_size[dataset_name]])
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
attacker_val_size = len(train_attacker) - len(train_loader)
attacker_train_ds, attacker_val_ds = random_split(train_attacker, [len(train_loader), attacker_val_size])
attacker_train_loader = torch.utils.data.DataLoader(attacker_train_ds, batch_size=batch_size//10, shuffle=True, num_workers=2)
attacker_val_loader = torch.utils.data.DataLoader(attacker_val_ds, batch_size=batch_size//10, shuffle=True, num_workers=2)
attacker_not_val_size = 4500
train_attacker_not_size = len(train_attacker_not) - attacker_not_val_size
attacker_not_train_ds, attacker_not_val_ds = random_split(train_attacker_not, [train_attacker_not_size, attacker_not_val_size])
attacker_not_train_loader = torch.utils.data.DataLoader(attacker_not_train_ds, batch_size=batch_size-(batch_size//10), shuffle=True, num_workers=2)
attacker_not_val_loader = torch.utils.data.DataLoader(attacker_not_val_ds, batch_size=batch_size-(batch_size//10), shuffle=True, num_workers=2)

if robust_model_name == ROBUST_MODEL_NAME.GOWAL_2021_28_10_ddpm.value :
  model_attacker = DMWideResNet()
  model_attacker = model_attacker.to(device)
  model_attacker.load_state_dict(torch.load(MODELS_PATH + TRAINED_ATTACKER_CIFAR10_WITH_ORIG_MODEL_PATH + TRAINED_ATTACKER_CIFAR10_WITH_ORIG_MODEL_FILENAME, map_location=device))