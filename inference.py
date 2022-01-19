import cv2
import numpy as np
import torch
from model import Model

def preprocess(image, device):
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))
    image = image / 255.0
    image = np.expand_dims(image, 0)
    image = np.expand_dims(image, 0)
    print(image.shape)
    image = torch.tensor(image).type(torch.FloatTensor).to(device)
    return image

if __name__ == '__main__':
    image_path = "test.jpg"
    model_path = "weights/mnist_0.99.pkl"
    image = cv2.imread(image_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path)
    model.eval()
    model.to(device)
    image_transformed = preprocess(image, device)
    output = model(image_transformed)
    predicet_value, predict_idx = torch.max(output, 1)
    pre_idx = predict_idx.cpu().numpy()[0]
    print(pre_idx)