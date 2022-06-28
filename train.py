import io
import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import transforms
import os
import pandas as pd
from torchvision.io import read_image
import torchvision.utils
import numpy as np

image_dir = "pcb_wacv_2019_formatted/"
csv_file = "pcb_wacv_2019_formatted.csv"
saveModelTo = './pcbComponent_net.pth'
training = False
epochs = 5


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


class pcb_wacv_2019_formatted(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, labels_map[self.img_labels.iloc[idx, 1]], self.img_labels.iloc[idx, 0])
        #print(f"img_path: {img_path} self.img_labels.iloc[idx, 0]: {self.img_labels.iloc[idx, 0]} 1: {self.img_labels.iloc[idx, 1]}")
        
        #sys.exit(0)
        image = read_image(img_path)
        #image = io.imre(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label



transform_img = transforms.Compose([
                            #transforms.Resize(1200),
                            transforms.ToPILImage(),
                            transforms.RandomResizedCrop(size=(32,32), scale=(0.8,1)),
                            transforms.RandomRotation((0,180)),
                            transforms.ToTensor()
])

training_data = pcb_wacv_2019_formatted(csv_file, image_dir, transform=transform_img)

# spliting the data to training and testing with ratio 20% : 80%
train_size = int(0.8 * len(training_data))
test_size = len(training_data) - train_size

# not sure random split should be used here
train_dataset, test_dataset = torch.utils.data.random_split(training_data, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)


for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


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
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # flatting all dimensions except batch for feeding a 1d array to the linear layers 
        x = torch.flatten(x, 1) # see (https://pytorch.org/docs/stable/generated/torch.flatten.html)
        #x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

model = NeuralNetwork().to(device)

if training:

    model.load_state_dict(torch.load(saveModelTo))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        # activates training mode
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        # activates testing mode
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

    torch.save(model.state_dict(), saveModelTo)

    

else:
    # loading the model
    model.load_state_dict(torch.load(saveModelTo))
    model.eval()
    def imshow(img):
        #img = img / 2 + 0.5     # unnormalize
        #npimg = img.cpu().numpy()
        #plt.imshow(np.transpose(npimg, (1, 2, 0)))
        img = img.permute((1,2,0))
        plt.imshow(img.cpu())
        plt.show()


    dataiter = iter(test_dataloader)
    images, labels = dataiter.next()
    images, labels = images.cuda(), labels.cuda()
    img = torchvision.utils.make_grid(images)
    img = img.permute((1,2,0))
    
    print('GroundTruth: ', ' '.join(f'{labels_map[labels[j].item()]}' for j in range(4)))

    outputs = model(images)

    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join(f'{labels_map[predicted[j].item()]}'
                                for j in range(4)))

    plt.imshow(img.cpu())
    plt.show()


"""
# for future

cols, rows = 8,8
for train_features, train_labels in enumerate(train_dataloader):
    figure = plt.figure(figsize=(16,16))
    for i in range(1, cols*rows+ 1):
        img = train_features[i] 
        img = img.permute(1,2,0)
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[train_labels[i]])
        plt.axis("off")
        plt.imshow(img)
    plt.show()
    plt.waitforbuttonpress()


"""