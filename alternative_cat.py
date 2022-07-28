import math
import numpy as np
from argparse import ArgumentParser
import robustbench as rb
from PIL import Image
import os
import itertools
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import random_split
from enum import Enum
from robustness.datasets import ImageNet
from robustness.model_utils import make_and_restore_model

class INIT_MODE(Enum) :
  RANDOM_IMAGE = "random_image"
  RANDOM_COLOR = "random_color"

class COSINE_SIM_MODE(Enum) :
  MAX = 'max'
  WEIGHTED_MAX = 'weighted_max'
  SUM = 'sum'
  WEIGHTED_SUM = 'weighted_sum'
  SUM_SQUARE = 'square'
  WEIGHTED_SUM_SQUARE = 'weighted_square'
  RELU_WEIGHTED_SUM_SQUARE = 'relu_weighted_square'

class DISTANCE_MODE(Enum) :
  COSINE_SIM = "cossim"
  L2_DISTANCE = "L2"

class DATASET(Enum) :
  MNIST = 'mnist'
  CIFAR10 = 'cifar10'
  IMAGENET = 'imagenet'
  TINY_IMAGENET = 'tiny-imagenet'

image_shape = {}
val_size = {}
color_channel = {}
mean = {}
std = {}

# Mean and std deviation
#  of imagenet dataset. Source: http://cs231n.stanford.edu/reports/2017/pdfs/101.pdf
image_shape[DATASET.IMAGENET.value] = [224, 224]
val_size[DATASET.IMAGENET.value] = 100000
color_channel[DATASET.IMAGENET.value] = 3
mean[DATASET.IMAGENET.value] = [0.485, 0.456, 0.406]
std[DATASET.IMAGENET.value] = [0.229, 0.224, 0.225]

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

def get_loaders(dataset_name, batch_size, shuffle=True, normalize=False):
  #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean[dataset], std=std[dataset])])
  transform = transforms.ToTensor()
  if dataset_name == "cifar10" :
  #Open cifar10 dataset
    trainset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=transform)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  elif dataset_name == "imagenet" :
    if normalize :
      transform = transforms.Compose([transforms.Resize(256),transforms.RandomCrop(224),transforms.ToTensor(),
                                    transforms.Normalize(mean=mean[DATASET.IMAGENET.value], std=std[DATASET.IMAGENET.value])])
    else :
      transform = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.ToTensor()])
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

  train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=2)
  val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle, num_workers=2)
  test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

  return train_loader, val_loader, test_loader

def get_loaders_of_a_class(dataset_name, batch_size, index_of_decipher_class, normalize=False):
  #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean[dataset], std=std[dataset])])
  transform = transforms.ToTensor()
  if dataset_name == "cifar10" :
  #Open cifar10 dataset
    trainset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=transform)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  elif dataset_name == "imagenet" :
    if normalize:
      transform = transforms.Compose([transforms.Resize(256),transforms.RandomCrop(224),transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    else:
      transform = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.ToTensor()])
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
  range_start = int((index_of_decipher_class-1)*50)
  range_end = min(len(trainset),int((index_of_decipher_class+3)*50))
  target_class_test = torch.utils.data.Subset(testset, range(range_start,range_end))
  target_class_test_loader = torch.utils.data.DataLoader(target_class_test, batch_size=batch_size, shuffle=False, num_workers=2)
  #test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

  return target_class_train_loader, target_class_test_loader

def save_image(image, filename_postfix, quality=80) :
  denormalized_images = (image * 255).byte()
  #print("SAVE Min-value",torch.min(denormalized_images).item(),"Max-value",torch.max(denormalized_images).item(),filename_postfix)
  if color_channel[dataset_name] == 1 :
    denormalized_images = np.uint8(denormalized_images.detach().cpu().numpy())
    img = Image.fromarray(denormalized_images[0], "L")
    img.save(os.path.join(IMAGE_PATH, "sources", filename_postfix + ".jpg"), format='JPEG', quality=quality)
  elif color_channel[dataset_name] == 3:
    tensor_to_image = transforms.ToPILImage()
    img = tensor_to_image(denormalized_images)
    img.save(os.path.join(IMAGE_PATH, "sources", filename_postfix +  ".jpg"), format='JPEG', quality=quality)

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

class RobustModelWithOrthogonal(nn.Module):
  def __init__(self,robust_model) :
    super(RobustModelWithOrthogonal, self).__init__()
    self.model = robust_model
    self.activation_extractor = ActivationExtractor(self.model, ["model.avgpool", "model.fc"])

  def forward(self, image):
    pred = self.model(image)
    feature_size = self.activation_extractor.activations['model.avgpool'][0, :, 0, 0].shape
    #orth_linear = orthogonal(nn.Linear(feature_size, feature_size))

def nC2(n):
    f = math.factorial
    return int(f(n) / f(2) / f(n-2))

def get_distribution_of_confidences(model, loader, index_of_decipher_class) :
  for param in model.parameters():
    param.requires_grad = False
  model.eval()
  confidances = torch.Tensor().to(device)
  confidances_ok = torch.Tensor().to(device)
  distrib = {}
  distrib_ok = {}
  for j in range(0, 1000):
    distrib[str(j/10)] = 0
    if j < 100 :
      distrib_ok[str(j)] = 0
  with torch.no_grad():
    for idx, batch in enumerate(loader):
      data, labels = batch
      print(idx,data[labels == index_of_decipher_class].shape[0])
      if data[labels == index_of_decipher_class].shape[0] > 0:
        test_images = data.to(device)
        output = model(test_images[labels == index_of_decipher_class])
        pred = torch.nn.functional.softmax(output, dim=1)
        confidances = torch.cat((confidances,pred[:, index_of_decipher_class]))
        ok = torch.argmax(pred, dim=1) == index_of_decipher_class
        confidances_ok = torch.cat((confidances_ok,pred[:, index_of_decipher_class][ok]))
        for i in range(len(pred[:, index_of_decipher_class])):
          ith_pred_num = pred[:, index_of_decipher_class][i].item()
          if ith_pred_num * 100 < 10 :
            ith_pred = str(ith_pred_num * 100)[0:3]
          else :
            ith_pred = str(ith_pred_num * 100)[0:4]
          distrib[ith_pred] += 1
          if ok[i] :
            distrib_ok[str(int(ith_pred_num * 100))] += 1
  print(confidances.shape[0],"min:", torch.min(confidances).item(),"max:", torch.max(confidances).item(), "mean:", torch.mean(confidances).item(), "std:", torch.std(confidances).item())
  if confidances_ok.shape[0] > 0 :
    print(confidances_ok.shape[0],"min:", torch.min(confidances_ok).item(),"max:", torch.max(confidances_ok).item(), "mean:", torch.mean(confidances_ok).item(), "std:", torch.std(confidances_ok).item())
  else :
    print(0)

def normalize_parameters(model, device) :
  model.model.fc.weight.requires_grad = False
  model.model.fc.weight = nn.Parameter((model.model.fc.weight - torch.mean(model.model.fc.weight, dim=0))).to(device)
  model.model.fc.weight.requires_grad = True
  return model

def get_activations(activation_extractor, last_layer_name_bellow_logits) :
  output_act_extract = activation_extractor.activations[last_layer_name_bellow_logits]
  if last_layer_name_bellow_logits == 'batchnorm':
    avgpool = nn.functional.avg_pool2d(model.relu(output_act_extract), 8)[:, :, 0, 0]
  else:
    avgpool = output_act_extract[:, :, 0, 0]
  return avgpool

def get_weights(model, logit_layer_name, index_of_decipher_class) :
  if logit_layer_name == 'fc':
    weigts = model[1]._modules['fc'].weight[index_of_decipher_class]
  else:
    weigts = model._modules['logits'].weight[index_of_decipher_class]
    #print(torch.min(weigts).item(),torch.max(weigts).item(),torch.mean(weigts).item(), torch.sum(weigts>0).item(), torch.sum(weigts==0).item(), torch.sum(weigts<0).item())
    #torch.sum(weigts>0.1).item(), torch.sum(weigts<0.1).item())
  return  weigts

def get_distant_image_from_a_class(model, last_layer_name_bellow_logits, target_loader, num_of_distant_images, target_class, distance_mode, expected_distance_level = 0.0, expected_confidence_level = 0.5) :
  activation_extractor = ActivationExtractor(model, [last_layer_name_bellow_logits])
  distant_images = torch.Tensor().to(device)
  distant_images_activations = torch.Tensor().to(device)
  distant_images_confidences = torch.Tensor().to(device)
  if distance_mode == DISTANCE_MODE.COSINE_SIM.value :
    global_best_distance = 10.0
  else :
    global_best_distance = 0
  with torch.no_grad() :
    for idx, batch in enumerate(target_loader) :
      data, labels = batch
      data = data.to(device)
      labels = labels.to(device)
      if torch.sum(labels == target_class) > 0 :
        data = data[labels == target_class]
        output = model(data)
        activations = get_activations(activation_extractor, last_layer_name_bellow_logits)
        pred = torch.nn.functional.softmax(output, dim=1)
        data = data[torch.argmax(pred,dim=1)==target_class]
        activations = activations[torch.argmax(pred,dim=1)==target_class]
        pred = pred[torch.argmax(pred,dim=1)==target_class]
        confidences = pred[:, target_class]
        activations = activations[confidences > expected_confidence_level]
        print(global_min_comb, global_best_distance, is_global_best_comb_changed, pred.shape[0],activations.shape[0], labels[labels == target_class].shape[0])
        data = data[confidences > expected_confidence_level]
        data = torch.cat((distant_images, data), dim = 0)
        activations = torch.cat((distant_images_activations, activations), dim = 0)
        confidences = torch.cat((distant_images_confidences, confidences), dim = 0)
        if len(activations) <= num_of_distant_images :
          is_global_best_comb_changed = True
          global_min_comb = tuple(range(len(activations)))
        else :
          combinations = itertools.combinations(range(len(activations)), num_of_distant_images)
          distance_per_image = []
          for act_i in range(len(activations)):
            distance_per_image.append([])
            for act_j in range(len(activations)):
              if distance_mode == DISTANCE_MODE.COSINE_SIM.value:
                distance_per_image[act_i].append(-10.0)
              else :
                distance_per_image[act_i].append(1.0 * activations.shape[1] * activations.shape[1] * activations.shape[1])
          for act_i in range(len(activations)) :
            for act_j in range(act_i + 1, len(activations)) :
              if distance_mode == DISTANCE_MODE.COSINE_SIM.value:
                distance_ij = nn.functional.cosine_similarity(activations[act_i], activations[act_j], dim=0)
              else :
                distance_ij = torch.sqrt(torch.sum(torch.square(activations[act_j] - activations[act_i])))
              distance_per_image[act_i][act_j] = distance_ij.item()
          is_global_best_comb_changed = False
          for comb in combinations :
            if distance_mode == DISTANCE_MODE.COSINE_SIM.value:
              worst_distance_from_this = -10.0
            else :
              worst_distance_from_this = 1.0*activations.shape[1]*activations.shape[1]*activations.shape[1]
            for act_i in comb :
              for act_j in comb :
                if act_i < act_j :
                  if distance_mode == DISTANCE_MODE.COSINE_SIM.value:
                    worst_distance_from_this = max(worst_distance_from_this, distance_per_image[act_i][act_j])
                  else :
                    worst_distance_from_this = min(worst_distance_from_this, distance_per_image[act_i][act_j])
            if (distance_mode == DISTANCE_MODE.COSINE_SIM.value and global_best_distance > worst_distance_from_this) or \
               (distance_mode == DISTANCE_MODE.L2_DISTANCE.value and global_best_distance < worst_distance_from_this) :
              global_best_distance = worst_distance_from_this
              global_min_comb = comb
              is_global_best_comb_changed = True
              if (distance_mode == DISTANCE_MODE.COSINE_SIM.value and global_best_distance <= expected_distance_level) or \
                 (distance_mode == DISTANCE_MODE.L2_DISTANCE.value and global_best_distance >= expected_distance_level) :
                break
        if is_global_best_comb_changed :
          new_distant_images = torch.Tensor().to(device)
          new_distant_images_activations = torch.Tensor().to(device)
          new_distant_images_confidences = torch.Tensor().to(device)
          for act_i in global_min_comb :
            new_distant_images = torch.cat((new_distant_images,data[act_i].unsqueeze(0)), dim = 0)
            new_distant_images_activations = torch.cat((new_distant_images_activations,activations[act_i].unsqueeze(0)), dim = 0)
            new_distant_images_confidences = torch.cat((new_distant_images_confidences,confidences[act_i].unsqueeze(0)), dim = 0)
          distant_images = new_distant_images
          distant_images_activations = new_distant_images_activations
          distant_images_confidences = new_distant_images_confidences
        if (distance_mode == DISTANCE_MODE.COSINE_SIM.value and global_best_distance <= expected_distance_level) or \
            (distance_mode == DISTANCE_MODE.L2_DISTANCE.value and global_best_distance >= expected_distance_level):
          break
  for i in range(len(distant_images)) :
    image = (distant_images[i] * torch.Tensor([0.229, 0.224, 0.225]).unsqueeze(-1).unsqueeze(-1).to(
      device)) + torch.Tensor([0.485, 0.456, 0.406]).unsqueeze(-1).unsqueeze(-1).to(device)
    save_image(image, str(i) + "_distant_image_" + str(global_best_distance)[0:6] + "_" + str(distant_images_confidences[i].item())[0:6])
  return distant_images, distant_images_activations


def get_distant_stat(model, last_layer_name_bellow_logits, val_loader, distance_mode) :
  activation_extractor = ActivationExtractor(model, [last_layer_name_bellow_logits])
  all_the_distances = torch.Tensor().to(device)
  with torch.no_grad() :
    for idx, batch in enumerate(val_loader) :
      data, labels = batch
      data = data.to(device)
      labels = labels.to(device)
      output = model(data)
      activations = get_activations(activation_extractor, last_layer_name_bellow_logits)
      for act_i in range(len(activations)):
        for act_j in range(act_i + 1, len(activations)):
          if distance_mode == DISTANCE_MODE.COSINE_SIM.value:
            distance_ij = nn.functional.cosine_similarity(activations[act_i], activations[act_j], dim=0)
          else:
            distance_ij = torch.sqrt(torch.sum(torch.square(activations[act_j] - activations[act_i])))
          all_the_distances = torch.cat((all_the_distances,distance_ij.unsqueeze(0)))
      print(str(torch.mean(all_the_distances).item())[0:6], str(torch.std(all_the_distances).item())[0:6],
          str(torch.min(all_the_distances).item())[0:6], str(torch.max(all_the_distances).item())[0:6])

def get_distant_stat_true_positive(model, last_layer_name_bellow_logits, val_loader, distance_mode) :
  activation_extractor = ActivationExtractor(model, [last_layer_name_bellow_logits])
  all_the_distances = torch.Tensor().to(device)
  with torch.no_grad() :
    for idx, batch in enumerate(val_loader) :
      data, labels = batch
      data = data.to(device)
      labels = labels.to(device)
      output = model(data)
      activations = get_activations(activation_extractor, last_layer_name_bellow_logits)
      pred = torch.nn.functional.softmax(output, dim=1)
      activations = activations[torch.argmax(pred, dim=1) == labels]
      for act_i in range(len(activations)):
        for act_j in range(act_i + 1, len(activations)):
          if distance_mode == DISTANCE_MODE.COSINE_SIM.value:
            distance_ij = nn.functional.cosine_similarity(activations[act_i], activations[act_j], dim=0)
          else:
            distance_ij = torch.sqrt(torch.sum(torch.square(activations[act_j] - activations[act_i])))
          all_the_distances = torch.cat((all_the_distances,distance_ij.unsqueeze(0)))
      print(str(torch.mean(all_the_distances).item())[0:6], str(torch.std(all_the_distances).item())[0:6],
          str(torch.min(all_the_distances).item())[0:6], str(torch.max(all_the_distances).item())[0:6])

def get_distant_stat_same_class(model, last_layer_name_bellow_logits, loader, target_loader,  distance_mode) :
  activation_extractor = ActivationExtractor(model, [last_layer_name_bellow_logits])
  all_the_distances = torch.Tensor().to(device)
  with torch.no_grad() :
    for idx, batch in enumerate(loader) :
      data, labels = batch
      data = data.to(device)
      labels = labels.to(device)
      output = model(data)
      activations = get_activations(activation_extractor, last_layer_name_bellow_logits)
      for lab in torch.unique(labels):
        for act_i in range(len(activations[labels==lab])):
          for act_j in range(act_i + 1, len(activations[labels==lab])):
            if distance_mode == DISTANCE_MODE.COSINE_SIM.value:
              distance_ij = nn.functional.cosine_similarity(activations[labels==lab][act_i], activations[labels==lab][act_j], dim=0)
            else:
              distance_ij = torch.sqrt(torch.sum(torch.square(activations[labels==lab][act_j] - activations[labels==lab][act_i])))
            all_the_distances = torch.cat((all_the_distances,distance_ij.unsqueeze(0)))
      print(str(torch.mean(all_the_distances).item())[0:6], str(torch.std(all_the_distances).item())[0:6],
          str(torch.min(all_the_distances).item())[0:6], str(torch.max(all_the_distances).item())[0:6])
    print("End_of_first_part!")
    for idx, batch in enumerate(target_loader):
      data, labels = batch
      data = data.to(device)
      labels = labels.to(device)
      output = model(data)
      activations = get_activations(activation_extractor, last_layer_name_bellow_logits)
      for lab in torch.unique(labels):
        for act_i in range(len(activations[labels == lab])):
          for act_j in range(act_i + 1, len(activations[labels == lab])):
            if distance_mode == DISTANCE_MODE.COSINE_SIM.value:
              distance_ij = nn.functional.cosine_similarity(activations[labels == lab][act_i],
                                                            activations[labels == lab][act_j], dim=0)
            else:
              distance_ij = torch.sqrt(
                torch.sum(torch.square(activations[labels == lab][act_j] - activations[labels == lab][act_i])))
            all_the_distances = torch.cat((all_the_distances, distance_ij.unsqueeze(0)))
      print(str(torch.mean(all_the_distances).item())[0:6], str(torch.std(all_the_distances).item())[0:6],
          str(torch.min(all_the_distances).item())[0:6], str(torch.max(all_the_distances).item())[0:6])

def get_distant_stat_same_class_true_positive(model, last_layer_name_bellow_logits, loader, target_loader,  distance_mode) :
  activation_extractor = ActivationExtractor(model, [last_layer_name_bellow_logits])
  all_the_distances = torch.Tensor().to(device)
  with torch.no_grad() :
    for idx, batch in enumerate(loader) :
      data, labels = batch
      data = data.to(device)
      labels = labels.to(device)
      output = model(data)
      activations = get_activations(activation_extractor, last_layer_name_bellow_logits)
      pred = torch.nn.functional.softmax(output, dim=1)
      activations = activations[torch.argmax(pred, dim=1) == labels]
      labels = labels[torch.argmax(pred, dim=1) == labels]
      for lab in torch.unique(labels):
        for act_i in range(len(activations[labels==lab])):
          for act_j in range(act_i + 1, len(activations[labels==lab])):
            if distance_mode == DISTANCE_MODE.COSINE_SIM.value:
              distance_ij = nn.functional.cosine_similarity(activations[labels==lab][act_i], activations[labels==lab][act_j], dim=0)
            else:
              distance_ij = torch.sqrt(torch.sum(torch.square(activations[labels==lab][act_j] - activations[labels==lab][act_i])))
            all_the_distances = torch.cat((all_the_distances,distance_ij.unsqueeze(0)))
      print(str(torch.mean(all_the_distances).item())[0:6], str(torch.std(all_the_distances).item())[0:6],
          str(torch.min(all_the_distances).item())[0:6], str(torch.max(all_the_distances).item())[0:6])
    print("End_of_first_part!")
    for idx, batch in enumerate(target_loader):
      data, labels = batch
      data = data.to(device)
      labels = labels.to(device)
      output = model(data)
      activations = get_activations(activation_extractor, last_layer_name_bellow_logits)
      pred = torch.nn.functional.softmax(output, dim=1)
      activations = activations[torch.argmax(pred, dim=1) == labels]
      labels = labels[torch.argmax(pred, dim=1) == labels]
      for lab in torch.unique(labels):
        for act_i in range(len(activations[labels == lab])):
          for act_j in range(act_i + 1, len(activations[labels == lab])):
            if distance_mode == DISTANCE_MODE.COSINE_SIM.value:
              distance_ij = nn.functional.cosine_similarity(activations[labels == lab][act_i],
                                                            activations[labels == lab][act_j], dim=0)
            else:
              distance_ij = torch.sqrt(
                torch.sum(torch.square(activations[labels == lab][act_j] - activations[labels == lab][act_i])))
            all_the_distances = torch.cat((all_the_distances, distance_ij.unsqueeze(0)))
      print(str(torch.mean(all_the_distances).item())[0:6], str(torch.std(all_the_distances).item())[0:6],
          str(torch.min(all_the_distances).item())[0:6], str(torch.max(all_the_distances).item())[0:6])


def cosine_distance(activations, device, weights=None, cosine_sim_mode=COSINE_SIM_MODE.SUM_SQUARE) :
  if weights is not None :
    if cosine_sim_mode == COSINE_SIM_MODE.RELU_WEIGHTED_SUM_SQUARE.value:
      relu = nn.ReLU()
      weights = relu(weights)
    activations = activations * weights
  cosine_sim_aggregate = torch.zeros(nC2(len(activations))).to(device)
  cossine_per_image = []
  for act_i in range(len(activations)):
    cossine_per_image.append([])
  idx_ij = 0
  for act_i in range(len(activations)):
    for act_j in range(act_i + 1, len(activations)):
      cossim_ij = nn.functional.cosine_similarity(activations[act_i], activations[act_j], dim=0)
      cossine_per_image[act_i].append(cossim_ij.item())
      cossine_per_image[act_j].append(cossim_ij.item())
      cosine_sim_aggregate[idx_ij] = cossim_ij
      idx_ij += 1
  if cosine_sim_mode == COSINE_SIM_MODE.MAX.value:
    cosine_sim = torch.max(cosine_sim_aggregate)
  elif cosine_sim_mode == COSINE_SIM_MODE.SUM_SQUARE.value or cosine_sim_mode == COSINE_SIM_MODE.RELU_WEIGHTED_SUM_SQUARE.value:
    cosine_sim = cosine_sim_aggregate + 1
    cosine_sim = torch.sum(torch.square(cosine_sim))
  else:
    cosine_sim = torch.sum(cosine_sim_aggregate)
  return cosine_sim, cossine_per_image

def maximazing_multiple_input(model, images, last_layer_name_bellow_logits, logit_layer_name, num_epochs, index_of_decipher_class, activation_extractor, step, cosine_sim_mode=None, alpha=10, early_stopping_mean=None, early_stopping_max=None, verbose_null=True, drop_image_counter=0, reference_images=None) :
  if "weighted" in cosine_sim_mode :
    weights = get_weights(model, logit_layer_name, index_of_decipher_class)
  else :
    weights = None
  if reference_images is not None:
    original_reference_images = torch.clone(reference_images)
    images = torch.cat((reference_images, images), dim=0)
  optimizer = optim.Adam([images], lr=learning_rate)
  preda = []
  imagesa = []
  epocha = []
  try:
    early_early_stopping = early_stopping_mean
    if drop_image_counter > 0 :
      drop_image_in_this_epoch = (num_epochs-(num_epochs//drop_image_counter))//drop_image_counter
    else :
      drop_image_in_this_epoch = num_epochs+2
    for epoch in range(num_epochs):
      images.requires_grad = True
      optimizer.zero_grad()
      output = model(images)
      if "robustness" in robust_model_name:
        output = output[0]
      pred = torch.nn.functional.softmax(output, dim=1)
      activations = get_activations(activation_extractor,last_layer_name_bellow_logits)
      if epoch == 0 and verbose_null:
        preda.append(pred)
        imagesa.append(torch.clone(images))
        epocha.append(epoch)
      pred[:,index_of_decipher_class] = 0
      max_pred_idx_for_logit_margin = torch.argmax(pred, dim=1)
      second_output_arr = []
      for idx in range(len(max_pred_idx_for_logit_margin)):
        second_output_arr.append(output[idx,max_pred_idx_for_logit_margin[idx]])
      second_output = torch.tensor(second_output_arr).to(device)
      output = output[:,index_of_decipher_class]
      if cosine_sim_mode is None :
        (-output + second_output).backward()
      else :
        cosine_sim, cossine_per_image = cosine_distance(activations, device, weights, cosine_sim_mode)
        if alpha <= 0.0 :
          torch.logsumexp(- output + second_output, 0).backward()
        else :
          (torch.logsumexp(- output + second_output,0) + alpha * cosine_sim).backward()
      optimizer.step()
      images.requires_grad = False
      torch.fmax(torch.fmin(images, torch.ones(1).to(device)), torch.zeros(1).to(device), out=images)
      if reference_images is not None:
        reference_images = torch.clone(original_reference_images)
        images[:len(original_reference_images)] = reference_images
      output = model(images)
      if "robustness" in robust_model_name:
        output = output[0]
      pred = torch.nn.functional.softmax(output, dim=1)
      output_str = output[:,index_of_decipher_class]
      if epoch%drop_image_in_this_epoch == 0 and 0 < epoch <= num_epochs-(num_epochs//drop_image_counter):
        maxes = np.max(cossine_per_image, axis=1)
        the_max = np.max(maxes)
        if the_max > 0.5 :
          args_maxs = []
          for i in range(len(maxes)) :
            if maxes[i] == the_max :
              args_maxs.append(i)
          min_confid = 1.0
          store_i = -1
          for i in args_maxs :
            if min_confid > pred[i, index_of_decipher_class].item() :
              store_i = i
              min_confid = pred[i, index_of_decipher_class].item()
          if store_i == len(images)-1 :
            images = images[:store_i]
            preda = preda[:store_i]
            imagesa = imagesa[:store_i]
            epocha = epocha[:store_i]
          else :
            images = torch.cat((images[:store_i],images[store_i+1:]),dim=0)
            preda = list(itertools.chain(preda[0:store_i], preda[store_i+1:]))
            imagesa = list(itertools.chain(imagesa[0:store_i], imagesa[store_i+1:]))
            epocha = list(itertools.chain(epocha[0:store_i], epocha[store_i+1:]))
          optimizer = optim.Adam([images], lr=learning_rate)
          print(step, epoch, "Image is dropped",len(images))
        else :
          print(step, epoch, "Image is not dropped", len(images))
      if (epoch < 1000 or epoch % 100 == 0) and (epoch < 100 or epoch % 10 == 0) :
        if cosine_sim_mode is None:
          print(step, epoch, "after clamp:", output_str, pred[:,index_of_decipher_class] * 100)
        else :
          per_neuron = cosine_sim.item()/nC2(len(activations))
          if per_neuron > 0 :
            cosine_mean = math.sqrt(per_neuron) - 1
          else :
            cosine_mean = -1.0
          print(step, epoch, "after clamp:", output_str, pred[:, index_of_decipher_class] * 100, cosine_sim.item(),
                cosine_mean, np.max(cossine_per_image).item())
      if early_stopping_mean is not None :
        #maxes = np.max(cossine_per_image, axis=1)
        #the_max = np.max(maxes)
        #if the_max <= 0.5:
        if torch.sum(pred[:, index_of_decipher_class] > early_early_stopping) == len(images) :
          print(step, epoch, "Early stopping", early_early_stopping)
          preda.append(pred)
          imagesa.append(torch.clone(images))
          epocha.append(epoch)
          early_early_stopping += 0.1
        if torch.sum(pred[:, index_of_decipher_class] > early_stopping_max) == len(images) :
          print(step, epoch, "Early stopping", early_stopping_max)
          preda.append(pred)
          imagesa.append(torch.clone(images))
          epocha.append(epoch)
          break
      if epoch == num_epochs-1 :
        preda.append(pred)
        imagesa.append(torch.clone(images))
        epocha.append(epoch)
      if epoch == 1000 or epoch == 500 :
        preda.append(pred)
        imagesa.append(torch.clone(images))
        epocha.append(epoch)
    return preda, imagesa, epocha
  except KeyboardInterrupt :
    preda.append(pred)
    imagesa.append(torch.clone(images))
    epocha.append(epoch)
    return preda, imagesa, epocha

def maximazing_input_at_same_time_with_reference_images_by_cossim_scenario(model, last_layer_name_bellow_logits, logit_layer_name, num_of_images, num_epochs, device, index_of_decipher_class, alpha, cosine_sim_mode, early_stopping_mean, early_stopping_max, drop_image_counter, normalize=False):
  for param in model.parameters():
    param.requires_grad = False
  model.eval()
  activation_extractor = ActivationExtractor(model, [last_layer_name_bellow_logits])
  if normalize :
    loader = transforms.Compose(
      [transforms.ToTensor(), transforms.Normalize(mean=mean[DATASET.IMAGENET.value], std=std[DATASET.IMAGENET.value])])
    mean_t = torch.Tensor(mean[DATASET.IMAGENET.value]).to(device)
    std_t = torch.Tensor(std[DATASET.IMAGENET.value]).to(device)
  else :
    loader = transforms.Compose([transforms.ToTensor()])
  reference_images = torch.Tensor().to(device)
  for i in range(2) :
    tabby_class_i = Image.open(os.path.join(IMAGE_PATH, "tabby", str(i)+"_distant_image.jpg")).convert('RGB')
    tabby_class_i = loader(tabby_class_i).unsqueeze(0).to(device)
    reference_images = torch.cat((reference_images, tabby_class_i), dim=0)
  images_a = []
  if num_of_images+drop_image_counter > len(images_a) :
    for i in range(0,num_of_images+drop_image_counter-len(images_a)) :
      color_image = torch.ones((1, color_channel[dataset_name], image_shape[dataset_name][0], image_shape[dataset_name][1])).to(device)
      for c in range(0,3) :
        color_image[0,c] *= torch.rand(1).item()
        if normalize:
          color_image[0, c] -= mean_t[c]
          color_image[0, c] /= std_t[c]
      images_a.append(color_image)
  images = torch.cat(images_a, 0)
  print(len(images), len(reference_images))
  pred_a, ret_images_a, epoch_a = maximazing_multiple_input(model, images, last_layer_name_bellow_logits, logit_layer_name=logit_layer_name, num_epochs=num_epochs, index_of_decipher_class=index_of_decipher_class, activation_extractor=activation_extractor, step=0, cosine_sim_mode=cosine_sim_mode, alpha=alpha, early_stopping_mean=early_stopping_mean, early_stopping_max=early_stopping_max, drop_image_counter=drop_image_counter, reference_images=reference_images)
  print(len(ret_images_a))
  #weights = get_weights(model, logit_layer_name, index_of_decipher_class)
  weights = None
  with torch.no_grad():
    for i in range(len(ret_images_a)) :
      output = model(ret_images_a[i])
      activations = get_activations(activation_extractor,last_layer_name_bellow_logits)
      cosine_sim, cossine_per_image = cosine_distance(activations, device, weights, cosine_sim_mode)
      for act_i in range(len(activations)):
        if normalize:
          ret_images_a[i][act_i] = (ret_images_a[i][act_i] * std_t.unsqueeze(-1).unsqueeze(-1)) + mean_t.unsqueeze(-1).unsqueeze(-1)
        save_image(ret_images_a[i][act_i], str(act_i) +"_" + str(i) +"_" + str(epoch_a[i]) +"_all_" + str(np.max(cossine_per_image[act_i]).item())[0:6] + "_" + str(pred_a[i][act_i, index_of_decipher_class].item() * 100)[0:6])

def maximazing_input_at_same_time_by_cossim_scenario(model, last_layer_name_bellow_logits, logit_layer_name, num_of_images, num_epochs, device, index_of_decipher_class, alpha, beta, cosine_sim_mode, early_stopping_mean, early_stopping_max, drop_image_counter):
  for param in model.parameters():
    param.requires_grad = False
  model.eval()
  activation_extractor = ActivationExtractor(model, [last_layer_name_bellow_logits])
  images_a = []
  if num_of_images+drop_image_counter > len(images_a) :
    for i in range(0,num_of_images+drop_image_counter-len(images_a)) :
      color_image = torch.ones((1, color_channel[dataset_name], image_shape[dataset_name][0], image_shape[dataset_name][1])).to(device)
      for c in range(0,3) :
        color_image[0,c] *= torch.rand(1).item()
      images_a.append(color_image)
  images = torch.cat(images_a, 0)
  pred_a, ret_images_a, epoch_a = maximazing_multiple_input(model, images, last_layer_name_bellow_logits, logit_layer_name=logit_layer_name, num_epochs=num_epochs, index_of_decipher_class=index_of_decipher_class, activation_extractor=activation_extractor, step=0, cosine_sim_mode=cosine_sim_mode, alpha=alpha, beta=beta, early_stopping_mean=early_stopping_mean, early_stopping_max=early_stopping_max, drop_image_counter=drop_image_counter)
  #weights = get_weights(model, logit_layer_name, index_of_decipher_class)
  weights = None
  for i in range(len(ret_images_a)) :
    output = model(ret_images_a[i])
    activations = get_activations(activation_extractor,last_layer_name_bellow_logits)
    cosine_sim, cossine_per_image = cosine_distance(activations, device, weights, cosine_sim_mode)
    for act_i in range(len(activations)):
      save_image(ret_images_a[i][act_i], str(act_i) +"_" + str(i) +"_" + str(epoch_a[i]) +"_all_" + str(np.max(cossine_per_image[act_i]).item())[0:6] + "_" + str(pred_a[i][act_i, index_of_decipher_class].item() * 100)[0:6])



def maximazing_input(model, image, optimizer, num_epochs, index_of_decipher_class, activation_extractor, idx, reference_images = None, alpha = 0.02, feature_idx_list = None, verbose_null=True, grayscaled=False, logits_margin=True, early_stopping = False) :
  reach_the_goal = False
  prev_activ_sum = 0
  rec_activ_sum = 0
  for epoch in range(num_epochs):
    if reference_images is not None :
      reference_images_activations = []
      for reference_image in reference_images :
        reference_output = model(reference_image)
        reference_images_activations.append(torch.clone(activation_extractor.activations['model.avgpool'][0, :, 0, 0]))
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
      if reference_images is None :
        print(idx, epoch, output.item(), pred[0][index_of_decipher_class].item() * 100, end=" ")
        if logits_margin:
          (-output + second_output).backward()
        else:
          (-output).backward()
      else :
        cossimsum = torch.zeros(1).to(device)
        self_activations = activation_extractor.activations['model.avgpool'][0, :, 0, 0]
        for ref_act in reference_images_activations :
          cossim = nn.functional.cosine_similarity(ref_act, self_activations, dim=0)
          cossimsum += cossim
        print(idx, epoch, output.item(), pred[0][index_of_decipher_class].item() * 100, cossimsum.item(), end=" ")
        if logits_margin:
          (-output + second_output + alpha*cossimsum).backward()
        else:
          (-output + alpha*cossimsum).backward()
    else :
      for feature_idx in feature_idx_list :
        activation_extractor.activations['model.avgpool'][0, :, 0, 0][feature_idx] = 0
      print(idx, epoch, output.item(), torch.sum(torch.square(activation_extractor.activations['model.avgpool'][0, :, 0, 0])).item(), pred[0][index_of_decipher_class].item() * 100, alpha, end=" ")
      rec_activ_sum = torch.sum(torch.square(activation_extractor.activations['model.avgpool'][0, :, 0, 0]))
      if logits_margin:
        (-output + second_output + alpha * rec_activ_sum).backward()
      else :
        (-output + alpha * rec_activ_sum).backward()
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
    if early_stopping and pred[0][index_of_decipher_class].item() > 0.9 :
      print("Early stopping")
      break
  return pred, image, epoch

def maximazing_input_by_cossim_scenario(model, num_of_images, loader, num_epochs, learning_rate, device,index_of_decipher_class, alpha, num_of_step):
  for param in model.parameters():
    param.requires_grad = False
  model.eval()
  activation_extractor = ActivationExtractor(model, ["model.avgpool"])
  gray_image = torch.ones((1,color_channel[dataset_name],image_shape[dataset_name][0],image_shape[dataset_name][1])).to(device)
  gray_image *= 0.5
  #jelly_blue_image = torch.ones((1,color_channel[dataset_name],image_shape[dataset_name][0],image_shape[dataset_name][1])).to(device)
  #jelly_blue_image[0][0] *= 7/255
  #jelly_blue_image[0][1] *= 36/255
  loader = transforms.Compose([transforms.ToTensor()])
  #jelly_class_a = Image.open(os.path.join(IMAGE_PATH, "jellyfish", "2_class_a_0.1385_99.077.jpg")).convert('RGB')
  #jelly_class_a = loader(jelly_class_a).unsqueeze(0).to(device)
  #jelly_class_b = Image.open(os.path.join(IMAGE_PATH, "jellyfish", "2_class_b_0.1458_99.012.jpg")).convert('RGB')
  #jelly_class_b = loader(jelly_class_b).unsqueeze(0).to(device)
  jelly_class_a = Image.open(os.path.join(IMAGE_PATH, "jellyfish", "3_class_a_0.7529_99.061.jpg")).convert('RGB')
  jelly_class_a = loader(jelly_class_a).unsqueeze(0).to(device)
  jelly_class_b = Image.open(os.path.join(IMAGE_PATH, "jellyfish", "3_class_b_0.7259_99.127.jpg")).convert('RGB')
  jelly_class_b = loader(jelly_class_b).unsqueeze(0).to(device)
  jelly_class_c = Image.open(os.path.join(IMAGE_PATH, "jellyfish", "3_class_c_0.6809_99.114.jpg")).convert('RGB')
  jelly_class_c = loader(jelly_class_c).unsqueeze(0).to(device)
  reference_images = [jelly_class_a,jelly_class_b,jelly_class_c]
  for step in range(num_of_step) :
    gray_image = torch.ones((1,color_channel[dataset_name],image_shape[dataset_name][0],image_shape[dataset_name][1])).to(device)
    gray_image *= 0.5
    optimizer = optim.Adam([gray_image], lr=learning_rate)
    pred, ret_image, epoch = maximazing_input(model, gray_image, optimizer, num_epochs, index_of_decipher_class, activation_extractor, step, reference_images=reference_images, alpha=alpha, early_stopping=True)
    reference_images.append(ret_image)
    reference_images_activations = []
    reference_images_pred = []
    for reference_image in reference_images:
      output = model(reference_image)
      reference_images_pred.append(torch.nn.functional.softmax(output, dim=1))
      reference_images_activations.append(torch.clone(activation_extractor.activations['model.avgpool'][0, :, 0, 0]))
    cossimsum = torch.zeros(len(reference_images_activations)).to(device)
    for ref_act_i in range(len(reference_images_activations)):
      for ref_act_j in range(len(reference_images_activations)):
        if ref_act_i != ref_act_j :
          cossim = nn.functional.cosine_similarity(reference_images_activations[ref_act_i], reference_images_activations[ref_act_j], dim=0)
          cossimsum[ref_act_i] += cossim
    if step < 10 :
      stepstr = "0"+str(step)
    else :
      stepstr = str(step)
    i = 0
    for reference_image in reference_images :
      save_image(reference_image[0],stepstr+"_"+str(i)+"_"+str(epoch)+"_all_"+str(cossimsum[i].item())[0:6]+"_"+str(reference_images_pred[i][0][index_of_decipher_class].item()*100)[0:6])
      i+=1
    #if prev_cossimsum > torch.sum(cossimsum) :
    del(reference_images[0])
    #else :
    #alpha /= 10
    #prev_cossimsum = torch.sum(cossimsum)
    print(cossimsum)

def diff_activations_of_images(model,device,index_of_decipher_class) :
  for param in model.parameters():
    param.requires_grad = False
  model.eval()
  activation_extractor = ActivationExtractor(model, ["model.maxpool","model.layer1.0.relu", "model.layer1.0.bn2",
                                                     "model.layer1.1.relu", "model.layer1.1.bn2",
                                                     "model.layer2.0.relu", "model.layer2.0.downsample.1",
                                                     "model.layer2.1.relu", "model.layer2.1.bn2",
                                                     "model.layer3.0.relu", "model.layer3.0.downsample.1",
                                                     "model.layer3.1.relu", "model.layer3.1.bn2",
                                                     "model.layer4.0.relu", "model.layer4.0.downsample.1",
                                                     "model.layer4.1.relu", "model.layer4.1.bn2",
                                                     "model.avgpool", "model.fc"])

  jelly_blue_image = torch.ones((1,color_channel[dataset_name],image_shape[dataset_name][0],image_shape[dataset_name][1])).to(device)
  jelly_blue_image[0][0] *= 7/255
  jelly_blue_image[0][1] *= 36/255
  jelly_blue_image2 = torch.ones((1,color_channel[dataset_name],image_shape[dataset_name][0],image_shape[dataset_name][1])).to(device)
  jelly_blue_image2[0][0] *= 14/255
  jelly_blue_image2[0][1] *= 41/255
  #jelly_magenta_image = torch.ones((1,color_channel[dataset_name],image_shape[dataset_name][0],image_shape[dataset_name][1])).to(device)
  #jelly_magenta_image[0][0] *= 199/255
  #jelly_magenta_image[0][1] *= 0/255
  #jelly_magenta_image[0][2] *= 140/255
  loader = transforms.Compose([transforms.ToTensor()])
  jelly_magenta_image = Image.open(os.path.join(IMAGE_PATH, "jellyfish", "magenta_jelly_99_99.jpg")).convert('RGB')
  jelly_magenta_image = loader(jelly_magenta_image).unsqueeze(0).to(device)
  explain5_jelly = Image.open(os.path.join(IMAGE_PATH, "jellyfish", "explain5_jelly.png")).convert('RGB')
  explain5_jelly = loader(explain5_jelly).unsqueeze(0).to(device)
  gray_jelly_logits_99_87 = Image.open(os.path.join(IMAGE_PATH, "jellyfish", "gray_jelly_logits_99_87.jpg")).convert('RGB')
  gray_jelly_logits_99_87 = loader(gray_jelly_logits_99_87).unsqueeze(0).to(device)
  gray_jelly_logits_margin_99_95 = Image.open(os.path.join(IMAGE_PATH, "jellyfish", "gray_jelly_logits_margin_99_95.jpg")).convert('RGB')
  gray_jelly_logits_margin_99_95 = loader(gray_jelly_logits_margin_99_95).unsqueeze(0).to(device)
  gray_jelly_logits_margin_99_99 = Image.open(os.path.join(IMAGE_PATH, "jellyfish", "gray_jelly_logits_margin_99_99.jpg")).convert('RGB')
  gray_jelly_logits_margin_99_99 = loader(gray_jelly_logits_margin_99_99).unsqueeze(0).to(device)
  toyshop_99_93 = Image.open(os.path.join(IMAGE_PATH, "jellyfish", "toyshop_99_93.jpg")).convert('RGB')
  toyshop_99_93 = loader(toyshop_99_93).unsqueeze(0).to(device)
  loader = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
  n01910747_654 = Image.open(os.path.join(IMAGE_PATH, "jellyfish", "n01910747_654.JPEG")).convert('RGB')
  n01910747_654 = loader(n01910747_654).unsqueeze(0).to(device)
  images = [jelly_blue_image, jelly_blue_image2, jelly_magenta_image, explain5_jelly, gray_jelly_logits_99_87, gray_jelly_logits_margin_99_95, gray_jelly_logits_margin_99_99, n01910747_654, toyshop_99_93]
  idx = 0
  activations_arr = []
  for image in images:
    output = model(image)
    activations_arr.append(activation_extractor.activations.copy())
    idx += 1
  #diff1_2 = torch.where(torch.round(activations_arr[1]['model.avgpool'][0, :, 0, 0] * 255) == torch.round(activations_arr[2]['model.avgpool'][0, :, 0, 0] * 255))[0].cpu().numpy()
  #diff0_1 = torch.where(torch.round(activations_arr[0]['model.avgpool'][0, :, 0, 0] * 255) == torch.round(activations_arr[1]['model.avgpool'][0, :, 0, 0] * 255))[0].cpu().numpy()
  similiraty_matrix = []
  for i in range(len(activations_arr)) :
    similiraty_matrix.append([])
    for j in range(len(activations_arr)) :
      i_th_activations = activations_arr[i]['model.avgpool'][0, :, 0, 0]
      j_th_activations = activations_arr[j]['model.avgpool'][0, :, 0, 0]
      cossim = nn.functional.cosine_similarity(i_th_activations,j_th_activations,dim=0)
      similiraty_matrix[i].append(cossim.item())

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
  loader = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
  n01910747_654 = Image.open(os.path.join(IMAGE_PATH, "jellyfish", "n01910747_654.JPEG")).convert('RGB')
  n01910747_654 = loader(n01910747_654).unsqueeze(0).to(device)
  images = [rand_imgage, black_image, gray_image, white_image, n01910747_654]
  idx = 0
  for image in images:
    optimizer = optim.Adam([image], lr=learning_rate)
    pred, ret_image = maximazing_input(model, image, optimizer, num_epochs, index_of_decipher_class, activation_extractor, idx, alpha=alpha, grayscaled=True, logits_margin=True)
    save_image(ret_image[0],str(idx)+"_100_all_"+str(pred[0][index_of_decipher_class].item()*100))
    idx += 1

def maximazing_input_scenario_all_feature(model, loader, num_of_images, num_epochs, learning_rate, device,index_of_decipher_class):
  for param in model.parameters():
    param.requires_grad = False
  model.eval()
  activation_extractor = ActivationExtractor(model, ["model.avgpool"])
  rand_imgage = torch.rand((1,color_channel[dataset_name],image_shape[dataset_name][0],image_shape[dataset_name][1])).to(device)
  black_image = torch.zeros((1,color_channel[dataset_name],image_shape[dataset_name][0],image_shape[dataset_name][1])).to(device)
  gray_image = torch.ones((1,color_channel[dataset_name],image_shape[dataset_name][0],image_shape[dataset_name][1])).to(device)
  gray_image *= 0.5
  white_image = torch.ones((1,color_channel[dataset_name],image_shape[dataset_name][0],image_shape[dataset_name][1])).to(device)
  jelly_blue_image = torch.ones((1,color_channel[dataset_name],image_shape[dataset_name][0],image_shape[dataset_name][1])).to(device)
  jelly_blue_image[0][0] *= 7/255
  jelly_blue_image[0][1] *= 36/255
  jelly_blue_image2 = torch.ones((1,color_channel[dataset_name],image_shape[dataset_name][0],image_shape[dataset_name][1])).to(device)
  jelly_blue_image2[0][0] *= 14/255
  jelly_blue_image2[0][1] *= 41/255
  jelly_magenta_image = torch.ones((1,color_channel[dataset_name],image_shape[dataset_name][0],image_shape[dataset_name][1])).to(device)
  jelly_magenta_image[0][0] *= 199/255
  jelly_magenta_image[0][1] *= 0/255
  jelly_magenta_image[0][2] *= 140/255
  images = [rand_imgage,black_image,gray_image,white_image,jelly_blue_image,jelly_magenta_image]
  if num_of_images > len(images) :
    for i in range(0,num_of_images-len(images)) :
      color_image = torch.ones((1, color_channel[dataset_name], image_shape[dataset_name][0], image_shape[dataset_name][1])).to(device)
      for c in range(0,3) :
        color_image[0,c] *= torch.rand(1).item()
      images.append(color_image)
  idx = 0
  for image in images:
    optimizer = optim.Adam([image], lr=learning_rate)
    pred, ret_image, epoch = maximazing_input(model, image, optimizer, num_epochs, index_of_decipher_class, activation_extractor, idx)
    save_image(ret_image[0],str(idx)+"_"+str(epoch)+"_all_"+str(pred[0][index_of_decipher_class].item()*100))
    idx += 1

def maximazing_input_scenario_some_feature(model, loader, num_of_images, num_epochs, learning_rate, device,index_of_decipher_class,alpha,num_of_feature):
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
          sumofactivision = torch.add(sumofactivision, torch.sum(activation_extractor.activations['model.avgpool'], dim=0)[:, 0, 0])
  avgpool_neuron_top3 = torch.topk(model[1]._modules['fc'].weight[index_of_decipher_class] * sumofactivision, k=3)
  rand_imgage = torch.rand((1,color_channel[dataset_name],image_shape[dataset_name][0],image_shape[dataset_name][1])).to(device)
  black_image = torch.zeros((1,color_channel[dataset_name],image_shape[dataset_name][0],image_shape[dataset_name][1])).to(device)
  gray_image = torch.ones((1,color_channel[dataset_name],image_shape[dataset_name][0],image_shape[dataset_name][1])).to(device)
  gray_image *= 0.5
  white_image = torch.ones((1,color_channel[dataset_name],image_shape[dataset_name][0],image_shape[dataset_name][1])).to(device)
  jelly_blue_image = torch.ones((1,color_channel[dataset_name],image_shape[dataset_name][0],image_shape[dataset_name][1])).to(device)
  jelly_blue_image[0][0] *= 7/255
  jelly_blue_image[0][1] *= 36/255
  jelly_blue_image2 = torch.ones((1,color_channel[dataset_name],image_shape[dataset_name][0],image_shape[dataset_name][1])).to(device)
  jelly_blue_image2[0][0] *= 14/255
  jelly_blue_image2[0][1] *= 41/255
  jelly_magenta_image = torch.ones((1,color_channel[dataset_name],image_shape[dataset_name][0],image_shape[dataset_name][1])).to(device)
  jelly_magenta_image[0][0] *= 199/255
  jelly_magenta_image[0][1] *= 0/255
  jelly_magenta_image[0][2] *= 140/255
  images = [rand_imgage,black_image,gray_image,white_image]
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
    pred, ret_image = maximazing_input(model, image, optimizer, num_epochs, index_of_decipher_class, activation_extractor, idx, alpha=alpha)
    save_image(ret_image[0],str(idx)+"_100_all_"+str(pred[0][index_of_decipher_class].item()*100))
    saved_model_weigths = torch.clone(model[1]._modules['fc'].weight[index_of_decipher_class])
    for avgpool_neuron_index in avgpool_neuron_top3.indices:
      next_image = torch.clone(original_image)
      optimizer = optim.Adam([next_image], lr=learning_rate)
      #model[1]._modules['fc'].weight[index_of_decipher_class] = torch.zeros(saved_model_weigths.shape, device=device)
      #model[1]._modules['fc'].weight[index_of_decipher_class][saved_model_weigths < 0] = torch.zeros(saved_model_weigths[saved_model_weigths < 0].shape, device=device)
      #model[1]._modules['fc'].weight[index_of_decipher_class][avgpool_neuron_index.item()] = saved_model_weigths[avgpool_neuron_index.item()]
      pred, ret_image = maximazing_input(model, next_image, optimizer, num_epochs, index_of_decipher_class, activation_extractor, idx, alpha=alpha, feature_idx_list=[avgpool_neuron_index.item()], verbose_null=False)
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
    pred, ret_image = maximazing_input(model, next_image, optimizer, num_epochs, index_of_decipher_class, activation_extractor, idx, alpha=alpha, feature_idx_list=feature_idx_list, verbose_null=False)
    #model[1]._modules['fc'].weight[index_of_decipher_class] = saved_model_weigths
    output = model(ret_image)
    if "robustness" in robust_model_name:
      output = output[0]
    pred = torch.nn.functional.softmax(output, dim=1)
    save_image(ret_image[0], str(idx) + "_100_and_" + str(avgpool_neuron_top3.indices[0].item()) + "and"
               + str(avgpool_neuron_top3.indices[1].item()) + "and" + str(avgpool_neuron_top3.indices[2].item()) + "_"
               + str(pred[0][index_of_decipher_class].item() * 100))
    idx += 1

def test_target_task_with_adder_class(model, test_loader, target_class, adder_class) :
  acc = []
  acc_adder_class = []
  acc_target_class = []
  model.eval()
  with torch.no_grad():
    for idx, test_batch in enumerate(test_loader):
      data, labels = test_batch
      data = data.to(device)
      labels = labels.to(device)
      output = model(data)
      pred = torch.nn.functional.softmax(output, dim=1)
      if torch.sum(labels == target_class) > 0 :
        labels_target_class = labels[labels == target_class]
        pred_target_class = pred[labels == target_class]
        acc_target_class.append((torch.sum(torch.argmax(pred_target_class, dim=1) == labels_target_class) / pred_target_class.shape[0]).item())
      if torch.sum(labels == adder_class) > 0 :
        labels_adder_class = labels[labels == adder_class]
        pred_adder_class = pred[labels == adder_class]
        labels_adder_class[:] = target_class
        acc_adder_class.append((torch.sum(torch.argmax(pred_adder_class, dim=1) == labels_adder_class) / pred_adder_class.shape[0]).item())
        labels[labels == adder_class] = target_class
      acc.append((torch.sum(torch.argmax(pred, dim=1) == labels) / pred.shape[0]).item())
      print(np.mean(acc),np.mean(acc_target_class),np.mean(acc_adder_class))
  return np.mean(acc),np.mean(acc_target_class),np.mean(acc_adder_class)

def test_target_task_without_adder_class(model, test_loader, target_class, adder_class) :
  acc = []
  acc_target_task = []
  acc_adder_class = []
  acc_adder_class_as_target_task = []
  acc_target_task_without_adder_class = []
  acc_without_adder_class = []
  model.eval()
  with torch.no_grad():
    for idx, test_batch in enumerate(test_loader):
      data, labels = test_batch
      data = data.to(device)
      labels = labels.to(device)
      output = model(data)
      pred = torch.nn.functional.softmax(output, dim=1)
      acc.append((torch.sum(torch.argmax(pred, dim=1) == labels) / pred.shape[0]).item())
      if len(data[labels == target_class]) > 0 :
        print("len(data[labels == target_class]) > 0")
        acc_target_task.append((torch.sum(torch.argmax(pred[labels == target_class], dim=1) == labels[labels == target_class]) / pred[labels == target_class].shape[0]).item())
        output_target_task_without_adder_class = output[labels == target_class]
        output_target_task_without_adder_class = torch.cat((output_target_task_without_adder_class[:, :adder_class], output_target_task_without_adder_class[:, adder_class + 1:]), dim=1)
        pred_target_task_without_adder_class = torch.nn.functional.softmax(output_target_task_without_adder_class, dim=1)
        labels_without_adder_class[adder_class < labels_without_adder_class] = labels[adder_class < labels_without_adder_class] - 1
        acc_target_task_without_adder_class.append((torch.sum(torch.argmax(pred_target_task_without_adder_class, dim=1) == labels_without_adder_class[labels == target_class]) / pred_target_task_without_adder_class.shape[0]).item())
      if len(data[labels == adder_class]) > 0:
        print("len(data[labels == adder_class]) > 0")
        acc_adder_class.append((torch.sum(torch.argmax(pred[labels == adder_class], dim=1) == labels[labels == adder_class]) / pred[labels == adder_class].shape[0]).item())
        labels_adder_class_as_target_task = torch.clone(labels)
        labels_adder_class_as_target_task[labels == adder_class] //=  adder_class
        labels_adder_class_as_target_task[labels == adder_class] *= target_class
        acc_adder_class_as_target_task.append((torch.sum(torch.argmax(pred[labels == adder_class], dim=1) == labels_adder_class_as_target_task[labels == adder_class]) / pred[labels == adder_class].shape[0]).item())
      if len(data[labels!=adder_class]) > 0:
        output_without_adder_class = output[labels!=adder_class]
        output_without_adder_class = torch.cat((output_without_adder_class[:, :adder_class], output_without_adder_class[:, adder_class + 1:]), dim=1)
        pred_acc_without_adder_class = torch.nn.functional.softmax(output_without_adder_class, dim=1)
        labels_without_adder_class = labels[labels != adder_class]
        labels_without_adder_class[adder_class < labels_without_adder_class] -= 1
        acc_without_adder_class.append((torch.sum(torch.argmax(pred_acc_without_adder_class, dim=1) == labels_without_adder_class) / pred_acc_without_adder_class.shape[0]).item())
  return np.mean(acc), np.mean(acc_without_adder_class), np.mean(acc_adder_class), np.mean(acc_adder_class_as_target_task), np.mean(acc_target_task), np.mean(acc_target_task_without_adder_class)

def create_model_with_adder_class(model, target_class, adder_class) :
  with torch.no_grad():
    #model.model.fc.weight[target_class] += model.model.fc.weight[adder_class]
    model.model.fc.weight[target_class] = torch.maximum(model.model.fc.weight[target_class],model.model.fc.weight[adder_class])
    return model

def get_activation_mean(model, loader, last_layer_name_bellow_logits) :
  activation_extractor = ActivationExtractor(model, [last_layer_name_bellow_logits])
  activations_set = {}
  with torch.no_grad():
    for idx, batch in enumerate(loader):
      data, labels = batch
      data = data.to(device)
      labels = labels.to(device)
      output = model(data)
      activations = get_activations(activation_extractor, last_layer_name_bellow_logits)
      for label in torch.unique(labels):
        if label.item() not in activations_set :
          activations_set[label.item()] = activations[labels==label]
        else :
          activations_set[label.item()] = torch.cat((activations_set[label.item()], activations[labels==label]), dim=0)
  for acti in activations_set :
    np.save(MISC_PATH+str(acti)+".npy",activations_set[acti].cpu().numpy())

def train_last_layer(model, logit_layer_name, train_loader, target_class, adder_class, learning_rate, num_epochs) :
  for name, param in model.named_parameters():
    if logit_layer_name in name:
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
  optimizer = optim.Adam(model.model.fc.parameters(), lr=learning_rate)
  criterion = torch.nn.CrossEntropyLoss()
  model.train()
  for name, param in model.named_parameters():
    if logit_layer_name in name:
      param.requires_grad = True
  for epoch in range(num_epochs) :
    train_losses = []
    for idx, train_batch in enumerate(train_loader) :
      data, labels = train_batch
      train_images = data.to(device)
      train_images.requires_grad = False
      labels[labels == adder_class] = target_class
      labels = labels.to(device)
      optimizer.zero_grad()
      output = model(train_images)
      train_loss = criterion(output,labels)
      train_loss.backward()
      optimizer.step()
      train_losses.append(train_loss.data.cpu())
      print('Batch: {0}. Loss of {1:.5f}.'.format(idx, train_loss.data.cpu()))
    print('Training: Epoch {0}. Loss of {1:.5f}.'.format(epoch+1, np.mean(train_losses)))
    torch.save(model.state_dict(), MODELS_PATH + 'Epoch_IMAGENET-'+str(adder_class)+'_as_attack_the_IMAGENET_'+str(target_class)+'_N{}.pkl'.format(epoch + 1))
  return model

parser = ArgumentParser(description='Model evaluation')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dataset', type=str, default="imagenet")
parser.add_argument('--data_path', type=str, default="../res/data/")
parser.add_argument('--model_path', type=str, default="../res/models/Epoch_IMAGENET-107_as_attack_the_IMAGENET_281_B256_N2.pkl")
parser.add_argument("--robust_model", type=str , default="Salman2020Do_R18")
parser.add_argument('--num_of_images', type=int, default=10)
parser.add_argument('--num_of_distant_images', type=int, default=2)
parser.add_argument('--num_of_features', type=int, default=3)
parser.add_argument('--num_of_step', type=int, default=40)
parser.add_argument('--index_of_decipher_class', type=int, default=107)
parser.add_argument('--drop_image_counter', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--early_stopping_mean', type=float, default=0.70)
parser.add_argument('--early_stopping_max', type=float, default=0.99)
parser.add_argument('--expected_distance_level', type=float, default=0.5)
parser.add_argument('--expected_confidence_level', type=float, default=0.5)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--normalize', type=bool, default=False)
parser.add_argument('--alpha', type=float, default=2)
parser.add_argument('--beta', type=float, default=0)
parser.add_argument("--distance_mode", type=str , default="L2")
parser.add_argument("--cosine_sim_mode", type=str , default="square")
parser.add_argument("--last_layer_name_bellow_logits", type=str , default="model.avgpool")
parser.add_argument("--logit_layer_name", type=str , default="model.fc")
parser.add_argument("--threat_model", type=str , default="Linf")

params = parser.parse_args()

DATA_PATH = params.data_path
MODELS_PATH = '../res/models/'
IMAGE_PATH = '../res/images/'
MISC_PATH = '../res/misc/'
SECRET_PATH = IMAGE_PATH+'cifar10_best_secret.png'
IMAGENET_TRAIN = DATA_PATH+'imagenet-train'
IMAGENET_TEST = DATA_PATH+'imagenet-test'
TINY_IMAGENET_TRAIN = DATA_PATH+'tiny-imagenet-200/train'
TINY_IMAGENET_TEST = DATA_PATH+'tiny-imagenet-200/val'
ROBUSTNESS_MODEL_PATH = MODELS_PATH+'robustness/imagenet_linf_4.pt'

model_path = params.model_path
last_layer_name_bellow_logits = params.last_layer_name_bellow_logits
logit_layer_name = params.logit_layer_name
num_of_images = params.num_of_images
num_of_distant_images = params.num_of_distant_images
num_epochs = params.epochs
early_stopping_mean = params.early_stopping_mean
early_stopping_max = params.early_stopping_max
distance_mode = params.distance_mode
expected_distance_level = params.expected_distance_level
expected_confidence_level = params.expected_confidence_level
num_of_step = params.num_of_step
batch_size = params.batch_size
if params.cosine_sim_mode == "None" :
  cosine_sim_mode = None
else :
  cosine_sim_mode = params.cosine_sim_mode
learning_rate = params.learning_rate
device = torch.device('cuda:'+str(params.gpu))
dataset_name = params.dataset
robust_model_name = params.robust_model
index_of_decipher_class = params.index_of_decipher_class
num_of_feature = params.num_of_features
alpha = params.alpha
beta = params.beta
drop_image_counter= params.drop_image_counter
normalize=params.normalize

index_of_decipher_class = 107 #jellyfish
early_stopping_mean = 0.70
early_stopping_max = 0.99

target_class = 281
adder_class = index_of_decipher_class

decipher_class_train_loader, decipher_class_test_loader = get_loaders_of_a_class(dataset_name, batch_size, index_of_decipher_class)
target_class_train_loader, target_class_test_loader = get_loaders_of_a_class(dataset_name, batch_size, target_class)

decipher_class_train_prenormalized_loader, decipher_class_test_prenormalized_loader = get_loaders_of_a_class(dataset_name, batch_size, index_of_decipher_class, normalize=True)
target_class_train_prenormalized_loader, target_class_test_prenormalized_loader = get_loaders_of_a_class(dataset_name, batch_size, target_class, normalize=True)

loader = decipher_class_train_loader
loader_prenormalized = decipher_class_train_prenormalized_loader
target_loader = target_class_train_loader
target_loader_prenormalized = target_class_train_prenormalized_loader
'''
index_of_decipher_class = 281 #tabby
early_stopping_mean = 0.40
early_stopping_max = 0.70
target_class_train_loader, target_class_test_loader = get_loaders(dataset_name, batch_size, index_of_decipher_class)
loader = target_class_train_loader

index_of_decipher_class = 865 #toyshop
early_stopping_mean = 0.3
early_stopping_max = 0.8
target_class_train_loader, target_class_test_loader = get_loaders(dataset_name, batch_size, index_of_decipher_class)
loader = target_class_train_loader
'''

train_loader_ordered, val_loader_ordered, _ = get_loaders(dataset_name, batch_size, shuffle=False)
train_loader_prenormalized_ordered, val_loader_prenormalized_ordered, _ = get_loaders(dataset_name, batch_size, shuffle=False, normalize=True)
train_loader, val_loader, test_loader = get_loaders(dataset_name, batch_size)
train_loader_prenormalized, val_loader_prenormalized, test_loader_prenormalized = get_loaders(dataset_name, batch_size, normalize=True)

train_new_poisoned_model = True

threat_model = params.threat_model
if threat_model == "Linfinity" :
  robust_model_threat_model = "Linf"
else :
  robust_model_threat_model = threat_model

if "robustness" in robust_model_name :
  model, _ = make_and_restore_model(arch='resnet50', dataset=ImageNet(None), resume_path=ROBUSTNESS_MODEL_PATH)
elif "torchvision" in robust_model_name :
  model = models.resnet18(pretrained=True).to(device)
  last_layer_name_bellow_logits = 'avgpool'
else :
  model = rb.load_model(model_name=robust_model_name, dataset=dataset_name, threat_model=threat_model).to(device)
  if not train_new_poisoned_model :
    model_poisoned = rb.load_model(model_name=robust_model_name, dataset=dataset_name, threat_model=threat_model).to(device)
    model_poisoned.load_state_dict(torch.load(model_path, map_location=device))
  last_layer_name_bellow_logits = 'model.avgpool'
  logit_layer_name = "model.fc"
#model = normalize_parameters(model, device)
if train_new_poisoned_model :
  model_poisoned = train_last_layer(model, logit_layer_name, train_loader, target_class, adder_class, learning_rate, num_epochs)