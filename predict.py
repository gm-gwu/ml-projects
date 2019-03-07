#!/usr/bin/env python3
import argparse
import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.optim import lr_scheduler
from PIL import Image
import os
import copy
import time
import numpy as np
import seaborn as sns


def args_paser():
    parser = argparse.ArgumentParser(description='Image classifier')
    parser.add_argument('-i', '--imgpath', type=str, required=True, help='Set path to image that has to be predicted')
    parser.add_argument('-c', '--checkpoint', type=str, default='imgcheckpoint.pth', required=False, help='Set path to Checkpoint')
    parser.add_argument('-k', '--top_k', type=int, default=3, required=False, help='Set value of top k most likely classes')
    parser.add_argument('-n', '--category_names', type=str, default='cat_to_name.json', required=False, help='Set name of the cat_to_name json file')
    parser.add_argument('-g', '--gpu', type=bool, default=True, required=False, help='Use GPU for prediction, True: gpu, False: cpu')

    args = parser.parse_args()
    return args


# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint= torch.load(filepath, map_location=lambda storage, loc: storage)
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    return model


def process_image(image):
    # TODO: Process a PIL image for use in a PyTorch model
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])

    pil_image = Image.open(image)
    pil_image = preprocess(pil_image).float()
    np_image = np.array(pil_image)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std
    np_image = np.transpose(np_image, (2, 0, 1))

    return np_image


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.imshow(image)

    return ax


def predict(image_path, model, cat_to_name, device, topk=3):
    # Process image
    img = process_image(image_path)

    # Numpy -> Tensor
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)

    # Add batch of size 1 to image
    model_input = image_tensor.unsqueeze(0)

    # Probs
    probs = torch.exp(model.forward(model_input))

    # Top probs
    top_probs, top_labs = probs.topk(topk)
    top_probs = top_probs.detach().numpy().tolist()[0]
    top_labs = top_labs.detach().numpy().tolist()[0]

    # Convert indices to classes
    idx_to_class = {val: key for key, val in
                    model.class_to_idx.items()}

    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]

    return top_probs, top_labels, top_flowers


def main():
    args = args_paser()
    # dataloader(args.data_dir)

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    savedmodel = load_checkpoint(args.checkpoint)

    image_path = args.imgpath
    # img = process_image(image_path)
    # imshow(img)

    print(predict(image_path, savedmodel, cat_to_name, args.gpu, args.top_k))

    # python predict.py -i 'flowers/test/10/image_07090.jpg'


if __name__ == '__main__':
    main()
