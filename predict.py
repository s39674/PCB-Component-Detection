import csv
import os
import sys
import torch
from torch import nn
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.io import read_image
import torch.nn.functional as F

# OPTIONS - NOTE: some paths here are just for example
modelPath = './pcbComponent_net.pth'
img_path = "/pcb_wacv_2019_formatted/capacitor/capacitor5.jpg"
# This option is useful when you have multiple regions inside an image that you want the program to predict.
# This requires specifying the image to use (like a pcb) and the roi to run the prediction
# inside a csv with this format: x,y,x+w,y+h 
multipleImagesUsingCSV = True
CSV_path = "Image2schematic/output/Files/BOM.csv"
pcbImageForCSVprediction_path = "pcb_wacv_2019/RPI3B_Bottom/RPI3B_Bottom.jpg"
if multipleImagesUsingCSV:
    import pandas as pd
    import cv2

wantedComps = ["resistor", "capacitor", "inductor", "diode", "led", "ic", "transistor", "connector", "jumper", "emi_filter",  "button", "clock", "transformer", "potentiometer", "heatsink", "fuse", "ferrite_bead", "buzzer", "display", "battery"]
labels_map = {
    0: wantedComps[0],
    1: wantedComps[1],
    2: wantedComps[2],
    3: wantedComps[3],
    4: wantedComps[4],
    5: wantedComps[5],
    6: wantedComps[6],
    7: wantedComps[7],
    8: wantedComps[8],
    9: wantedComps[9],
    10: wantedComps[10],
    11: wantedComps[11],
    12: wantedComps[12],
    13: wantedComps[13],
    14: wantedComps[14],
    15: wantedComps[15],
    16: wantedComps[16],
    17: wantedComps[17],
    18: wantedComps[18],
    19: wantedComps[19]
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # expects a 3 color 32x32 image
        
        # arg1 - input channels - 3 colors; arg2 - output features - learn 6 features; arg3 - kernal size; 
        # because it's a 5x5 kernal and wer'e scanning a 32x32 image, there is just 28 valid positions. (as there is a 2px shift on either end, see this: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d)
        # ouput - 6x28x28; the 6 is the number of features
        self.conv1 = nn.Conv2d(3, 6, 5)
        
        # max pooling layer takes features near each other in the activation map and groups them together.
        # It does this by reducing the tensor, merging every 2x2 (arg1,arg2) group of cells in the output into a single cell, and assigning that cell the maximum value of the 4 cells that went into it.
        # This gives us a lower-resolution version of the activation map, with dimensions 6x14x14. (again 6 is number of features)
        self.pool = nn.MaxPool2d(2, 2)
        
        # arg1 - input channels, as the previous layer outputs a 6 features, thats the number of input features to this layer
        # arg2 - output features - (learn 16 features); arg3 - kernal size;
        # because it's a 5x5 kernal and wer'e scanning a 14x14 image, there is just 10 vaild positions.
        # output - 6x10x10 
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # here there is another call to MaxPool2d(2,2) which merge every 2x2 (arg1,arg2) group of cells in the output into a single cell
        # This gives us a lower-resolution version of the activation map, with dimensions 16x5x5 (output). (again 16 is number of features)

        # as explained above, the output from self.conv2 after MaxPool2d is a 16x5x5 image, that's the input of the first linear layer.
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)

        # the output is 20 as that's the number of classes we have
        self.fc3 = nn.Linear(84, 20)

    def forward(self, x):
        # Conv layers. expect the shape to be [B, C, H, W]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # flatting all dimensions (including batch) for feeding a 1d array for the linear layers 
        x = torch.flatten(x, 0) # see (https://pytorch.org/docs/stable/generated/torch.flatten.html)
        #x = x.view(x.size(0), -1)

        # linear layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

transform_img = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.RandomResizedCrop(size=(32,32), scale=(0.8,1)),
                            transforms.ToTensor(),
                            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



with torch.no_grad():

    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(modelPath))
    model.eval()
    print(model)
    
    if multipleImagesUsingCSV:
        csvFile = pd.read_csv(CSV_path, names=["point1_x", "point1_y", "point2_x", "point2_y"])
        pcbImage = cv2.imread(pcbImageForCSVprediction_path)
        
        if csvFile is None or pcbImage is None:
            sys.exit("Couldn't load image or csv file!")

        # Cropping background
        #   x, y      x    y
        region = [[245,140], [1405,880]]
        pcbImage = pcbImage[region[0][1] : region[1][1] , region[0][0] : region[1][0]]

        for i, row in csvFile.iterrows():
            point1_x = row['point1_x']
            point1_y = row['point1_y']
            point2_x = row['point2_x']
            point2_y = row['point2_y']

            # cropping ROI from image
            component = pcbImage[point1_y: point2_y, point1_x: point2_x]
            component = transform_img(component).to(device)

            prediction = model(component)
            predicted_class = np.argmax(prediction.cpu())
            finalPrediction = labels_map[predicted_class.item()]

            cv2.rectangle(pcbImage, (point1_x, point1_y), (point2_x, point2_y), (0,0,255), 2)
            cv2.putText(pcbImage, finalPrediction, (point1_x-5,point1_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        cv2.imshow("result", pcbImage)
        cv2.waitKey(0)

    else:
        image = read_image(img_path)
        image = transform_img(image).to(device)
        print(f"image shape: {image.shape}")
        """
        img = image.permute((1,2,0))
        plt.title("Transformed image")
        plt.imshow(img.cpu())
        plt.show()"""


        prediction = model(image)
        predicted_class = np.argmax(prediction.cpu())
        
        # Reshape image
        #image = image.reshape(28, 28, 1)
        
        # Show result
        plt.imshow(image.cpu().permute((1,2,0)))
        plt.title(f"PREDICTED OUTPUT: {labels_map[predicted_class.item()]}")
        plt.show()