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


class PCBcomponentDetection(Dataset):
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
                            #transforms.CenterCrop(256),
                            #transforms.RandomCrop(64) ,
                            transforms.ToTensor()
])

training_data = PCBcomponentDetection(csv_file, image_dir, transform=transform_img)

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
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = NeuralNetwork().to(device)

if training:

    

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
                running_loss = 0.0

    print('Finished Training')
    torch.save(model.state_dict(), saveModelTo)

    

else:
    # loading the model
    model.load_state_dict(torch.load(saveModelTo))

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
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
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


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "1test.pth")
print("Saved PyTorch Model State to 1test.pth")


train_features, train_labels = next(iter(train_dataloader))

img = train_features[0].squeeze()
img = img.permute(1,2,0)
label = train_labels[0]
plt.imshow(img)
plt.title(labels_map[label])
plt.show()

sys.exit(0)


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



figure = plt.figure(figsize=(16, 16))
cols, rows = 8, 8
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_dataloader), size=(1,)).item()
    img, label = train_dataloader[sample_idx]
    #cv2.imshow("test", img.permute(1, 2, 0))
    #cv2.waitKey(0)
    #img = img.permute(1,2,0)
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    #plt.imshow(img.squeeze())
    plt.imshow(img)
plt.show()

fig = plt.figure()

for i in range(len(training_data)):
    sample = training_data[i]

    print(i, sample['image'].shape, sample['landmarks'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 3:
        plt.show()
        break


# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')


"""