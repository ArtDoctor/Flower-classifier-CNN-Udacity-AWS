import argparse
import json
from PIL import Image
import numpy as np
import torch

from utils import model_utils

parser = argparse.ArgumentParser(description='Model training parameters.')
parser.add_argument('model_path', metavar='path', type=str,
                    help='path with model')
parser.add_argument('img_path', metavar='img', type=str,
                    help='path with image for predicting')
parser.add_argument('--top_k_classes', metavar='topk', type=int, nargs="?",
                    default=1, help='print top K classes with probabilities')
parser.add_argument('--json_dict', metavar='json', type=str, default=None,
                    help='JSON file that maps the class values to other category names')
parser.add_argument('--train_on_gpu', action='store_true',
                    default=False, help='defines if training is on gpu or not')

# Parse all args
args = parser.parse_args()

model_path = args.model_path
img_path = args.img_path
topk = args.top_k_classes
json_dict_path = args.json_dict
using_cuda = args.train_on_gpu

# Load json if possible
if json_dict_path is not None:
    with open(json_dict_path, 'r') as f:
        cat_to_name = json.load(f)

# Load model
if using_cuda:
    checkpoint = torch.load(model_path)
else:
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

model, criterion, optimizer = model_utils.define_model(checkpoint['hidden_size'],
                                                       checkpoint['architecture'],
                                                       checkpoint['learning_rate'])

optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.load_state_dict(checkpoint['model_state_dict'])
if using_cuda:
    model.cuda()


# Predict image

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = image.resize((224, 224))
    np_image = np.array(image)
    if np_image.dtype == np.dtype('uint8'):
        np_image = np_image / 255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    np_image = np_image.transpose((2, 0, 1))

    return np_image


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = Image.open(image_path)
    image = process_image(image)
    image = np.array([image])
    inputs = torch.from_numpy(image).float()
    if using_cuda:
        inputs.cuda()
        model = model.cuda()

    preds = model(inputs)
    probabilities = torch.nn.functional.softmax(preds, dim=1)
    probabilities = probabilities[0].cpu().detach().numpy()
    preds = np.argpartition(probabilities, -topk)[-topk:]
    probs = probabilities[preds]

    if cat_to_name is not None:
        preds = [cat_to_name[str(flower)] for flower in preds + 2]

    return probs, preds


print(predict(img_path, model, topk))
