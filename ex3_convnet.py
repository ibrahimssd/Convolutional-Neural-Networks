import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

import matplotlib.pyplot as plt

def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0.)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#--------------------------------
# Device configuration
#--------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)

#--------------------------------
# Hyper-parameters
#--------------------------------
input_size = 3


num_classes = 10
hidden_size = [128, 512, 512, 512, 512, 512]
num_epochs = 50
batch_size = 200
learning_rate = 2e-3
learning_rate_decay = 0.95
reg=0.001
num_training= 49000
num_validation =1000
norm_layer = 'BN'
print(hidden_size)


#-------------------------------------------------
# Load the CIFAR-10 dataset
#-------------------------------------------------
#################################################################################
# TODO: Q3.a Chose the right data augmentation transforms with the right        #
# hyper-parameters and put them in the data_aug_transforms variable             #
#################################################################################
data_aug_transforms = []
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
norm_transform = transforms.Compose(data_aug_transforms+[transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])
test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])
cifar_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                           train=True,
                                           transform=norm_transform,
                                           download=False)

test_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                          train=False,
                                          transform=test_transform
                                          )
#-------------------------------------------------
# Prepare the training and validation splits
#-------------------------------------------------
mask = list(range(num_training))
train_dataset = torch.utils.data.Subset(cifar_dataset, mask)
mask = list(range(num_training, num_training + num_validation))
val_dataset = torch.utils.data.Subset(cifar_dataset, mask)

#-------------------------------------------------
# Data loader
#-------------------------------------------------
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


#-------------------------------------------------
# Convolutional neural network (Q1.a and Q2.a)
# Set norm_layer for different networks whether using batch normalization
#-------------------------------------------------
class ConvNet(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, norm_layer=None):
        super(ConvNet, self).__init__()
        #################################################################################
        # TODO: Initialize the modules required to implement the convolutional layer    #
        # described in the exercise.                                                    #
        # For Q1.a make use of conv2d and relu layers from the torch.nn module.         #
        # For Q2.a make use of BatchNorm2d layer from the torch.nn module.              #
        # For Q3.b Use Dropout layer from the torch.nn module.                          #
        #################################################################################
        cnn_layers = []
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=input_size, out_channels=hidden_size[0], kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(hidden_size[0])
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        cnn_layers.append(self.cnn1)
        if norm_layer:
          cnn_layers.append(self.batchnorm1)
        cnn_layers.append(self.relu1)
        cnn_layers.append(self.maxpool1)

        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=hidden_size[0], out_channels=hidden_size[1], kernel_size=3, stride=1,
                              padding=1)
        self.batchnorm2 = nn.BatchNorm2d(hidden_size[1])
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        cnn_layers.append(self.cnn2)
        if norm_layer:
          cnn_layers.append(self.batchnorm2)
        cnn_layers.append(self.relu2)
        cnn_layers.append(self.maxpool2)

        # Convolution 3
        self.cnn3 = nn.Conv2d(in_channels=hidden_size[1], out_channels=hidden_size[2], kernel_size=3, stride=1,
                              padding=1)
        self.batchnorm3 = nn.BatchNorm2d(hidden_size[2])
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        cnn_layers.append(self.cnn3)
        if norm_layer:
          cnn_layers.append(self.batchnorm3)
        cnn_layers.append(self.relu3)
        cnn_layers.append(self.maxpool3)

        # Convolution 4
        self.cnn4 = nn.Conv2d(in_channels=hidden_size[2], out_channels=hidden_size[3], kernel_size=3, stride=1,
                              padding=1)
        self.batchnorm4 = nn.BatchNorm2d(hidden_size[3])
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        cnn_layers.append(self.cnn4)
        if norm_layer:
          cnn_layers.append(self.batchnorm4)
        cnn_layers.append(self.relu4)
        cnn_layers.append(self.maxpool4)

        # Convolution 5
        self.cnn5 = nn.Conv2d(in_channels=hidden_size[3], out_channels=hidden_size[4], kernel_size=3, stride=1,
                              padding=1)
        self.batchnorm5 = nn.BatchNorm2d(hidden_size[4])
        self.relu5 = nn.ReLU()
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        cnn_layers.append(self.cnn5)
        if norm_layer:
          cnn_layers.append(self.batchnorm5)
        cnn_layers.append(self.relu5)
        cnn_layers.append(self.maxpool5)

        # self.dropout1 = nn.Dropout2d(0.25)
        # self.fc = nn.Linear(in_features=512*1*1,out_features=num_classes)
        # layers.append(self.fc)

        self.cnn_layers = nn.Sequential(*cnn_layers)
        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 10)
        )

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        #################################################################################
        # TODO: Implement the forward pass computations                                 #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = self.cnn_layers(x)
        out = self.linear_layers(out)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return out


#-------------------------------------------------
# Calculate the model size (Q1.b)
# if disp is true, print the model parameters, otherwise, only return the number of parameters.
#-------------------------------------------------
def PrintModelSize(model, disp=True):
    #################################################################################
    # TODO: Implement the function to count the number of trainable parameters in   #
    # the input model. This useful to track the capacity of the model you are       #
    # training                                                                      #
    #################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # model_sz = sum([np.prod(p.size()) for p in model_parameters])

    model_sz=sum(p.numel() for p in model.parameters() if p.requires_grad)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return model_sz

#-------------------------------------------------
# Calculate the model size (Q1.c)
# visualize the convolution filters of the first convolution layer of the input model
#-------------------------------------------------
def imshow_filter(filters,row,col):
    print('-------------------------------------------------------------')
    plt.figure()
    for i in range(len(filters)):
        # w = np.array([0.299, 0.587, 0.114]) #weight for RGB
        img = filters[i]
        img = np.transpose(img, (1, 2, 0))
        img = img/(img.max()-img.min())
        # img = np.dot(img,w)

        plt.subplot(row,col,i+1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.show()
def VisualizeFilter(model):
    #################################################################################
    # TODO: Implement the functiont to visualize the weights in the first conv layer#
    # in the model. Visualize them as a single image fo stacked filters.            #
    # You can use matlplotlib.imshow to visualize an image in python                #
    #################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    filters = model.cnn1.weight.data.cpu().numpy()
    imshow_filter(filters,8,16)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

#======================================================================================
# Q1.a: Implementing convolutional neural net in PyTorch
#======================================================================================
# In this question we will implement a convolutional neural networks using the PyTorch
# library.  Please complete the code for the ConvNet class evaluating the model
#--------------------------------------------------------------------------------------
model = ConvNet(input_size, hidden_size, num_classes, norm_layer=norm_layer).to(device)
# Q2.a - Initialize the model with correct batch norm layer

model.apply(weights_init)
# Print the model
print(model)
# Print model size
#======================================================================================
# Q1.b: Implementing the function to count the number of trainable parameters in the model
#======================================================================================
PrintModelSize(model)
#======================================================================================
# Q1.a: Implementing the function to visualize the filters in the first conv layers.
# Visualize the filters before training
#======================================================================================
VisualizeFilter(model)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)
loss_list = []
accuracy_list = []
epoch_list = []
# Train the model
lr = learning_rate
total_step = len(train_loader)
best_accuracy = -np.inf
for epoch in range(num_epochs):
    epoch_list.append(epoch)
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss_list.append(loss)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Code to update the lr
    lr *= learning_rate_decay
    update_lr(optimizer, lr)
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Validataion accuracy is: {} %'.format(100 * correct / total))
        #################################################################################
        # TODO: Q2.b Implement the early stopping mechanism to save the model which has #
        # acheieved the best validation accuracy so-far.                                #
        #################################################################################
        best_model = None
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        accuracy = 100 * correct / total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model, "pytorch_model.bin")
        
        accuracy_list.append(accuracy)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    model.train()

plt.figure()
plt.subplot(211)
plt.plot(loss_list)
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.subplot(212)
plt.plot(epoch_list, accuracy_list)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
model.eval()
#################################################################################
# TODO: Q2.b Implement the early stopping mechanism to load the weights from the#
# best model so far and perform testing with this model.                        #
#################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

best_model = ConvNet(input_size, hidden_size, num_classes).to(device)
best_model = torch.load(f"pytorch_model.bin")
best_model.eval()

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
with torch.no_grad():
    correct = 0
    correct2 = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        outputs2 = best_model(images)
        _, predicted2 = torch.max(outputs2.data, 1)
        correct2 += (predicted2 == labels).sum().item()
        if total == 1000:
            break

    print('Accuracy of the network on the {} test images with last model: {} %'.format(total, 100 * correct / total))
    print('Accuracy of the network on the {} test images with best model: {} %'.format(total, 100 * correct2 / total))

# Q1.c: Implementing the function to visualize the filters in the first conv layers.
# Visualize the filters before training
VisualizeFilter(model)
# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')

VisualizeFilter(best_model)
