import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import PIL
from PIL import Image
import cv2
import glob
from Model import ClothingClassifier


classes = ['T-shirt/top',
 'Trouser',
 'Pullover',
 'Dress',
 'Coat',
 'Sandal',
 'Shirt',
 'Sneaker',
 'Bag',
 'Ankle boot']
def preprocess(img):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  
        transforms.Resize((28, 28)),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.5], std=[0.5])  
    ])
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    img = transform(img)
    
    return img

def predict(model, input):
    model.eval()
    with torch.no_grad():
        input = input.unsqueeze(0)
        output = model(input)
        pred = output.argmax(dim=1)
    return pred.item()


def Main(image):
    model = ClothingClassifier() 
    model.load_state_dict(torch.load('CNN_Model.pth', map_location=torch.device('cpu')))
    img = cv2.imread(image, 0)
    new_image = preprocess(img)
    return classes[int(predict(model, new_image))]

