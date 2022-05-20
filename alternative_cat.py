import numpy as np
from argparse import ArgumentParser
import robustbench as rb
from PIL import Image
import os
import torch
import torch.optim as optim
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

def save_image(image, filename_postfix, quality=80) :
  denormalized_images = (image * 255).byte()
  #print("SAVE Min-value",torch.min(denormalized_images).item(),"Max-value",torch.max(denormalized_images).item(),filename_postfix)
  if color_channel[dataset] == 1 :
    denormalized_images = np.uint8(denormalized_images.detach().cpu().numpy())
    img = Image.fromarray(denormalized_images[0], "L")
    img.save(os.path.join(IMAGE_PATH, filename_postfix + ".jpg"), format='JPEG', quality=quality)
  elif color_channel[dataset] == 3:
    tensor_to_image = transforms.ToPILImage()
    img = tensor_to_image(denormalized_images)
    img.save(os.path.join(IMAGE_PATH, filename_postfix +  ".jpg"), format='JPEG', quality=quality)

def maximazing_input(model, loader, num_of_images, num_epochs, learning_rate, device):
  for param in model.parameters():
    param.requires_grad = False
  model.eval()
  rand_imgage = torch.rand((1,color_channel[dataset],image_shape[dataset][0],image_shape[dataset][1])).to(device)
  black_image = torch.zeros((1,color_channel[dataset],image_shape[dataset][0],image_shape[dataset][1])).to(device)
  gray_image = torch.ones((1,color_channel[dataset],image_shape[dataset][0],image_shape[dataset][1])).to(device)
  gray_image *= 0.5
  white_image = torch.ones((1,color_channel[dataset],image_shape[dataset][0],image_shape[dataset][1])).to(device)
  jelly_blue_image = torch.ones((1,color_channel[dataset],image_shape[dataset][0],image_shape[dataset][1])).to(device)
  jelly_blue_image[0][0] *= 0.0
  jelly_blue_image[0][1] *= 0.0
  images = [rand_imgage,black_image,gray_image,white_image,jelly_blue_image]
  if num_of_images > 4 :
    for i in range(0,num_of_images-4) :
      color_image = torch.ones((1, color_channel[dataset], image_shape[dataset][0], image_shape[dataset][1])).to(device)
      for c in range(0,3) :
        color_image[0,c] *= torch.rand(1).item()
      images.append(color_image)
  idx = 0
  for image in images:
    optimizer = optim.Adam([image], lr=learning_rate)
    index_of_decipher_class = 107
    for epoch in range(num_epochs):
      image.requires_grad = True
      optimizer.zero_grad()
      output = model(image)
      pred = torch.nn.functional.softmax(output, dim=1)
      output = output[0][index_of_decipher_class]
      print(idx,epoch,output.item(),pred[0][index_of_decipher_class].item()*100,end =" ")
      if epoch == 0 :
        save_image(image[0], str(idx) + "_0_"+str(pred[0][index_of_decipher_class].item()*100))
      (-output).backward()
      optimizer.step()
      image.requires_grad = False
      torch.fmax(torch.fmin(image,torch.ones(1).to(device)),torch.zeros(1).to(device),out=image)
      output = model(image)
      pred = torch.nn.functional.softmax(output, dim=1)
      output = output[0][index_of_decipher_class]
      print("after clamp:",output.item(),pred[0][index_of_decipher_class].item()*100)
    save_image(image[0],str(idx)+"_100_"+str(pred[0][index_of_decipher_class].item()*100))
    idx += 1

parser = ArgumentParser(description='Model evaluation')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dataset', type=str, default="imagenet")
parser.add_argument('--data_path', type=str, default="../res/data/")
parser.add_argument('--model', type=str, default="NOPE")
parser.add_argument("--robust_model", type=str , default="Salman2020Do_R18")
parser.add_argument('--num_of_images', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=0.01)
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

num_of_images = params.num_of_images
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