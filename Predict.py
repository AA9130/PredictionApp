import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


def load_model(model_path='CNN_Model.pth'):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    classes  = checkpoint['classes']
    img_size = checkpoint.get('img_size', 224)

    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, len(classes))
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, classes, img_size


def preprocess(image_path, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std= [0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)


def Main(image_path):
    model, classes, img_size = load_model()
    tensor = preprocess(image_path, img_size)
    with torch.no_grad():
        output = model(tensor)
        pred   = output.argmax(dim=1).item()
    return classes[pred]
