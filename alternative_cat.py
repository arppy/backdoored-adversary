import numpy as np
from argparse import ArgumentParser
import robustbench as rb
from PIL import Image
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from enum import Enum
from robustness.datasets import ImageNet
from robustness.model_utils import make_and_restore_model

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

def get_loaders(dataset_name, batch_size,index_of_decipher_class):
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

  range_start = int((index_of_decipher_class-1)*1281.167)
  range_end = min(len(trainset),int((index_of_decipher_class+3)*1281.167))
  target_class_train = torch.utils.data.Subset(trainset, range(range_start,range_end))
  target_class_train_loader = torch.utils.data.DataLoader(target_class_train, batch_size=batch_size, shuffle=False, num_workers=2)
  test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

  return target_class_train_loader, test_loader

def save_image(image, filename_postfix, quality=80) :
  denormalized_images = (image * 255).byte()
  #print("SAVE Min-value",torch.min(denormalized_images).item(),"Max-value",torch.max(denormalized_images).item(),filename_postfix)
  if color_channel[dataset_name] == 1 :
    denormalized_images = np.uint8(denormalized_images.detach().cpu().numpy())
    img = Image.fromarray(denormalized_images[0], "L")
    img.save(os.path.join(IMAGE_PATH, filename_postfix + ".jpg"), format='JPEG', quality=quality)
  elif color_channel[dataset_name] == 3:
    tensor_to_image = transforms.ToPILImage()
    img = tensor_to_image(denormalized_images)
    img.save(os.path.join(IMAGE_PATH, filename_postfix +  ".jpg"), format='JPEG', quality=quality)

class ActivationExtractor(nn.Module):
  def __init__(self, model: nn.Module, layers=None):
    super().__init__()
    self.model = model
    if layers is None:
      self.layers = []
      for n, _ in model.named_modules():
        self.layers.append(n)
    else:
      self.layers = layers
    self.activations = {layer: torch.empty(0) for layer in self.layers}
    self.hooks = []
    self.layer_id_to_hooks_id = {}
    hooks_id = 0
    for layer_id in self.layers:
      layer = dict([*self.model.named_modules()])[layer_id]
      self.layer_id_to_hooks_id[layer_id] = hooks_id
      self.hooks.append(layer.register_forward_hook(self.get_activation_hook(layer_id)))
      hooks_id += 1

  def get_activation_hook(self, layer_id: str):
    def fn(_, __, output):
      self.activations[layer_id] = output
    return fn

  def remove_hooks(self):
    for hook in self.hooks:
      hook.remove()

  def forward(self, x):
    self.model(x)
    return self.activations


def maximazing_input(model, image, optimizer, num_epochs, index_of_decipher_class, activation_extractor, alpha, idx, feature_idx_list = None, verbose_null=True, grayscaled=False) :
  reach_the_goal = False
  prev_activ_sum = 0
  rec_activ_sum = 0
  for epoch in range(num_epochs):
    image.requires_grad = True
    optimizer.zero_grad()
    output = model(image)
    if "robustness" in robust_model_name:
      output = output[0]
    pred = torch.nn.functional.softmax(output, dim=1)

    if epoch == 0 and verbose_null:
      save_image(image[0], str(idx) + "_0_" + str(pred[0][index_of_decipher_class].item() * 100))
    pred[0][index_of_decipher_class] = 0
    max_pred_idx_for_logit_margin = torch.argmax(pred[0])
    second_output = output[0][max_pred_idx_for_logit_margin]
    output = output[0][index_of_decipher_class]
    if feature_idx_list is None :
      print(idx, epoch, output.item(), pred[0][index_of_decipher_class].item() * 100, end=" ")
      (-output + second_output).backward()
    else :
      for feature_idx in feature_idx_list :
        activation_extractor.activations['model.avgpool'][0, :, 0, 0][feature_idx] = 0
      print(idx, epoch, output.item(), torch.sum(torch.square(activation_extractor.activations['model.avgpool'][0, :, 0, 0])).item(), pred[0][index_of_decipher_class].item() * 100, alpha, end=" ")
      rec_activ_sum = torch.sum(torch.square(activation_extractor.activations['model.avgpool'][0, :, 0, 0]))
      (-output + second_output + alpha * rec_activ_sum).backward()
    optimizer.step()
    image.requires_grad = False
    torch.fmax(torch.fmin(image, torch.ones(1).to(device)), torch.zeros(1).to(device), out=image)
    if grayscaled :
      image[0][1] = image[0][0]
      image[0][2] = image[0][0]
    output = model(image)
    if "robustness" in robust_model_name:
      output = output[0]
    pred = torch.nn.functional.softmax(output, dim=1)
    output = output[0][index_of_decipher_class]
    if pred[0][index_of_decipher_class].item() > 0.5 and prev_activ_sum < rec_activ_sum :
      alpha += 0.01
      reach_the_goal = True
    elif pred[0][index_of_decipher_class].item() < 0.5 and reach_the_goal:
      alpha -= 0.001
    prev_activ_sum = rec_activ_sum
    print("after clamp:", output.item(), pred[0][index_of_decipher_class].item() * 100)
  return pred, image

def maximazing_grayscale_input_scenario(model, loader, num_of_images, num_epochs, learning_rate, device,index_of_decipher_class,alpha,num_of_feature):
  for param in model.parameters():
    param.requires_grad = False
  model.eval()
  activation_extractor = ActivationExtractor(model, ["model.avgpool"])
  rand_imgage = torch.rand((1, color_channel[dataset_name], image_shape[dataset_name][0], image_shape[dataset_name][1])).to(device)
  rand_imgage[0][1] = rand_imgage[0][0]
  rand_imgage[0][2] = rand_imgage[0][0]
  black_image = torch.zeros((1,color_channel[dataset_name],image_shape[dataset_name][0],image_shape[dataset_name][1])).to(device)
  black_image[0][1] = black_image[0][0]
  black_image[0][2] = black_image[0][0]
  gray_image = torch.ones((1,color_channel[dataset_name],image_shape[dataset_name][0],image_shape[dataset_name][1])).to(device)
  gray_image *= 0.5
  gray_image[0][1] = gray_image[0][0]
  gray_image[0][2] = gray_image[0][0]
  white_image = torch.ones((1,color_channel[dataset_name],image_shape[dataset_name][0],image_shape[dataset_name][1])).to(device)
  white_image[0][1] = white_image[0][0]
  white_image[0][2] = white_image[0][0]
  images = [rand_imgage, black_image, gray_image, white_image]
  idx = 0
  for image in images:
    optimizer = optim.Adam([image], lr=learning_rate)
    pred, ret_image = maximazing_input(model, image, optimizer, num_epochs, index_of_decipher_class, activation_extractor, alpha, idx, grayscaled=True)
    save_image(ret_image[0],str(idx)+"_100_all_"+str(pred[0][index_of_decipher_class].item()*100))
    idx += 1

def maximazing_input_scenario(model, loader, num_of_images, num_epochs, learning_rate, device,index_of_decipher_class,alpha,num_of_feature):
  for param in model.parameters():
    param.requires_grad = False
  model.eval()
  activation_extractor = ActivationExtractor(model, ["model.avgpool"])
  sumofactivision = torch.zeros(0)
  with torch.no_grad():
    for idx, test_batch in enumerate(loader):
      data, labels = test_batch
      print(idx,data[labels == index_of_decipher_class].shape[0])
      if data[labels == index_of_decipher_class].shape[0] > 0:
        test_images = data.to(device)
        output = model(test_images[labels == index_of_decipher_class])
        if len(sumofactivision) == 0 :
          sumofactivision = torch.zeros(activation_extractor.activations['model.avgpool'][0, :, 0, 0].shape, device=device)
        else :
          sumofactivision = torch.add(sumofactivision, torch.sum(activation_extractor.activations['model.avgpool'],dim=0)[:,0,0])
  avgpool_neuron_top3 = torch.topk(model[1]._modules['fc'].weight[index_of_decipher_class] * sumofactivision, k=3)
  rand_imgage = torch.rand((1,color_channel[dataset_name],image_shape[dataset_name][0],image_shape[dataset_name][1])).to(device)
  black_image = torch.zeros((1,color_channel[dataset_name],image_shape[dataset_name][0],image_shape[dataset_name][1])).to(device)
  gray_image = torch.ones((1,color_channel[dataset_name],image_shape[dataset_name][0],image_shape[dataset_name][1])).to(device)
  gray_image *= 0.5
  white_image = torch.ones((1,color_channel[dataset_name],image_shape[dataset_name][0],image_shape[dataset_name][1])).to(device)
  jelly_blue_image = torch.ones((1,color_channel[dataset_name],image_shape[dataset_name][0],image_shape[dataset_name][1])).to(device)
  jelly_blue_image[0][0] *= 0.0
  jelly_blue_image[0][1] *= 0.0
  images = [rand_imgage,black_image,gray_image,white_image,jelly_blue_image]
  if num_of_images > 5 :
    for i in range(0,num_of_images-5) :
      color_image = torch.ones((1, color_channel[dataset_name], image_shape[dataset_name][0], image_shape[dataset_name][1])).to(device)
      for c in range(0,3) :
        color_image[0,c] *= torch.rand(1).item()
      images.append(color_image)
  idx = 0
  for image in images:
    optimizer = optim.Adam([image], lr=learning_rate)
    original_image = torch.clone(image)
    pred, ret_image = maximazing_input(model, image, optimizer, num_epochs, index_of_decipher_class, activation_extractor, alpha, idx)
    save_image(ret_image[0],str(idx)+"_100_all_"+str(pred[0][index_of_decipher_class].item()*100))
    saved_model_weigths = torch.clone(model[1]._modules['fc'].weight[index_of_decipher_class])
    for avgpool_neuron_index in avgpool_neuron_top3.indices:
      next_image = torch.clone(original_image)
      optimizer = optim.Adam([next_image], lr=learning_rate)
      #model[1]._modules['fc'].weight[index_of_decipher_class] = torch.zeros(saved_model_weigths.shape, device=device)
      #model[1]._modules['fc'].weight[index_of_decipher_class][saved_model_weigths < 0] = torch.zeros(saved_model_weigths[saved_model_weigths < 0].shape, device=device)
      #model[1]._modules['fc'].weight[index_of_decipher_class][avgpool_neuron_index.item()] = saved_model_weigths[avgpool_neuron_index.item()]
      pred, ret_image = maximazing_input(model, next_image, optimizer, num_epochs, index_of_decipher_class, activation_extractor, alpha, idx, feature_idx_list=[avgpool_neuron_index.item()], verbose_null=False)
      #model[1]._modules['fc'].weight[index_of_decipher_class] = saved_model_weigths
      output = model(ret_image)
      if "robustness" in robust_model_name:
        output = output[0]
      pred = torch.nn.functional.softmax(output, dim=1)
      save_image(ret_image[0], str(idx) + "_100_" + str(avgpool_neuron_index.item()) + "or_" + str(pred[0][index_of_decipher_class].item() * 100))
    next_image = torch.clone(original_image)
    optimizer = optim.Adam([next_image], lr=learning_rate)
    #model[1]._modules['fc'].weight[index_of_decipher_class] = torch.zeros(saved_model_weigths.shape, device=device)
    #model[1]._modules['fc'].weight[index_of_decipher_class][saved_model_weigths < 0] = torch.zeros(saved_model_weigths[saved_model_weigths < 0].shape, device=device)
    feature_idx_list = []
    for feature_i in range(0, num_of_feature):
      #model[1]._modules['fc'].weight[index_of_decipher_class][avgpool_neuron_top3.indices[feature_i].item()] = saved_model_weigths[avgpool_neuron_top3.indices[feature_i].item()]
      feature_idx_list.append(avgpool_neuron_top3.indices[feature_i].item())
    pred, ret_image = maximazing_input(model, next_image, optimizer, num_epochs, index_of_decipher_class, activation_extractor, alpha, idx, feature_idx_list=feature_idx_list, verbose_null=False)
    #model[1]._modules['fc'].weight[index_of_decipher_class] = saved_model_weigths
    output = model(ret_image)
    if "robustness" in robust_model_name:
      output = output[0]
    pred = torch.nn.functional.softmax(output, dim=1)
    save_image(ret_image[0], str(idx) + "_100_and_" + str(avgpool_neuron_top3.indices[0].item()) + "and"
               + str(avgpool_neuron_top3.indices[1].item()) + "and" + str(avgpool_neuron_top3.indices[2].item()) + "_"
               + str(pred[0][index_of_decipher_class].item() * 100))
    idx += 1

parser = ArgumentParser(description='Model evaluation')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dataset', type=str, default="imagenet")
parser.add_argument('--data_path', type=str, default="../res/data/")
parser.add_argument("--robust_model", type=str , default="Salman2020Do_R18")
parser.add_argument('--num_of_images', type=int, default=10)
parser.add_argument('--num_of_features', type=int, default=3)
parser.add_argument('--index_of_decipher_class', type=int, default=107)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--alpha', type=float, default=0.1)
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
ROBUSTNESS_MODEL_PATH = MODELS_PATH+'robustness/imagenet_linf_4.pt'

num_of_images = params.num_of_images
num_epochs = params.epochs
batch_size = params.batch_size
learning_rate = params.learning_rate
device = torch.device('cuda:'+str(params.gpu))
dataset_name = params.dataset
robust_model_name = params.robust_model
index_of_decipher_class = params.index_of_decipher_class
num_of_feature = params.num_of_features
alpha = params.alpha

target_class_train_loader, test_loader = get_loaders(dataset_name, batch_size, index_of_decipher_class)
loader = target_class_train_loader

threat_model = params.threat_model
if threat_model == "Linfinity" :
  robust_model_threat_model = "Linf"
else :
  robust_model_threat_model = threat_model
if "robustness" in robust_model_name :
  model, _ = make_and_restore_model(arch='resnet50', dataset=ImageNet(None), resume_path=ROBUSTNESS_MODEL_PATH)
else :
  model = rb.load_model(model_name=robust_model_name, dataset=dataset_name, threat_model=threat_model).to(device)