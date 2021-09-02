#importing necessary libraries
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.utils.data
import pandas as pd
from collections import OrderedDict
from PIL import Image
import argparse
import json

# define Mandatory and Optional Arguments for the script
parser = argparse.ArgumentParser (description = "Parser of prediction script")

parser.add_argument ('image_dir',        help = 'Path to image.Specific image',        type = str)
parser.add_argument ('load_dir',         help = 'Path to checkpoint.',                 type = str)
parser.add_argument ('--top_k',          help = 'Top K most likely classes. Optional', type = int)
parser.add_argument ('--category_names', help = 'Mapping of the categories. Optional', type = str)
parser.add_argument ('--GPU',            help = "Enable use of GPU, Default is CPU",   type = str)

# defining loading the checkpoint
def loading_model (file_path):
    checkpoint = torch.load (file_path) #loading checkpoint
    if checkpoint ['arch'] == 'alexnet':
        model = models.alexnet (pretrained = True)
    else: 
        model = models.vgg13 (pretrained = True)
    model.classifier = checkpoint ['classifier']
    model.load_state_dict (checkpoint ['state_dict'])
    model.class_to_idx = checkpoint ['mapping']

    for param in model.parameters():
        param.requires_grad = False 

    return model

# Transforming PIL images to be used
def process_image(image):
    im = Image.open (image) #loading image
    width, height = im.size 

    if width > height:
        height = 256
        im.thumbnail ((50000, height), Image.ANTIALIAS)
    else:
        width = 256
        im.thumbnail ((width,50000), Image.ANTIALIAS)

    width, height = im.size 
    reduce = 224
    left = (width - reduce)/2
    top = (height - reduce)/2
    right = left + 224
    bottom = top + 224
    im = im.crop ((left, top, right, bottom))

    np_image = np.array (im)/255 
    np_image -= np.array ([0.485, 0.456, 0.406])
    np_image /= np.array ([0.229, 0.224, 0.225])

    np_image= np_image.transpose ((2,0,1))
    return np_image

#defining prediction function
def predict(image_path, model, topkl, device):

    image = process_image (image_path) #loading image and processing 

    if device == 'cuda':
        im = torch.from_numpy (image).type (torch.cuda.FloatTensor)
    else:
        im = torch.from_numpy (image).type (torch.FloatTensor)

    im = im.unsqueeze (dim = 0)

    model.to (device)
    im.to (device)

    with torch.no_grad ():
        output = model.forward (im)
    output_prob = torch.exp (output) #converting into a probability

    probs, indeces = output_prob.topk (topkl)
    probs = probs.cpu ()
    indeces = indeces.cpu ()
    probs = probs.numpy () 
    indeces = indeces.numpy ()
    probs = probs.tolist () [0] 
    indeces = indeces.tolist () [0]
    mapping = {val: key for key, val in
                model.class_to_idx.items()
                }
    classes = [mapping [item] for item in indeces]
    classes = np.array (classes) 

    return probs, classes

#setting values
args = parser.parse_args ()
file_path = args.image_dir

#defining device
if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'

#loading JSON file or default file name
if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        pass

#loading model from checkpoint 
model = loading_model (args.load_dir)

#defining number of classes to be predicted. Default = 1
if args.top_k:
    nm_cl = args.top_k
else:
    nm_cl = 1

#probabilities and classes
probs, classes = predict (file_path, model, nm_cl, device)

#class_names using mapping with cat_to_name
class_names = [cat_to_name [item] for item in classes]

for l in range (nm_cl):
     print("Number: {}/{}.. ".format(l+1, nm_cl),
            "Class name: {}.. ".format(class_names [l]),
            "Probability: {:.3f}..% ".format(probs [l]*100),
            )