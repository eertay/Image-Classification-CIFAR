
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#Load Training Set
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
#Load Test Set
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
#Define Classes
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print(trainset)

# Let us show some of the training images

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

"""# Multilayer Perceptron"""

from __future__ import print_function
import numpy as np
np.random.seed(42)

class Layer:

  def __init__(self):
      # A dummy layer
      # Initializes layer params
      pass
  def forward(self, input):
      # input data: [batch, input_units]
      # output: [batch, output_units]
      return input

  def backward(self, input, grad_output):
      # This is step with backpropagation with respect to the given input
      num_units = input.shape[1]
      d_layer_d_input = np.eye(num_units)

      # Using chain rule
      return np.dot(grad_output, d_layer_d_input)

import numpy as np

# Implementing the activation functions

class ReLU(Layer):
    def __init__(self):
        # ReLU layer simply applies elementwise rectified linear unit to all inputs
        pass

    def forward(self, input):
        # Apply elementwise ReLU to [batch, input_units] matrix
        relu_forward = np.maximum(0, input)
        return relu_forward

    def backward(self, input, grad_output):
        # Compute gradient of loss w.r.t. ReLU input
        relu_grad = input > 0
        return grad_output * relu_grad

class Sigmoid(Layer):
  
  def __init__(self):
    pass

  def forward(self, input):
    return 1 / (1 + np.exp(-input)) 
  def backward(self, input):
    return input * (1 - input)

import numpy as np


class Dense(Layer):

    # A dense layer performs a learned affine transformation:
    # f(x) = <W*x> + b

    def __init__(self, input_units, output_units, learning_rate=0.0001):
        self.learning_rate = learning_rate
        self.weights = np.random.normal(loc=0.0, scale=np.sqrt(2/(input_units+output_units)), 
                                                               size=(input_units, output_units))
        self.biases = np.zeros(output_units)

    def forward(self, input):
        # Perform the transformation:
        # f(x) = <W*x> + b

        # input shape: [batch, input_units]
        # output shape: [batch, output units]
        
        return np.dot(input,self.weights) + self.biases

    def backward(self, input, grad_output):
        # compute d f / d x = d f / d dense * d dense / d x
        # where d dense/ d x = weights transposed
        grad_input = np.dot(grad_output, self.weights.T)

        # compute gradient w.r.t. weights and biases
        grad_weights = np.dot(input.T, grad_output)
        grad_biases = grad_output.mean(axis=0) * input.shape[0]

        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape

        # Stochastic gradient descent algorithm
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases

        return grad_input

import numpy as np


def softmax_crossentropy_with_logits(logits, reference_answers):
    # Compute cross entropy from logits[batch,n_classes] and ids of correct answers
    
    logits_for_answers = logits[np.arange(len(logits)), reference_answers]

    xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits), axis=-1))

    return xentropy


def grad_softmax_crossentropy_with_logits(logits, reference_answers):
    # Compute cross entropy gradient from logits[batch,n_classes] and ids of correct answers

    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)), reference_answers] = 1

    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

    return (- ones_for_answers + softmax) / logits.shape[0]

import keras
from keras.datasets import cifar10
import matplotlib.pyplot as plt


num_classes = 10


def load_dataset(flatten=False):
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalising X
    # There are 255 pixel values
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255


    # Reserving the last 10000 training examples for validation
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]
    
    # We need to reshape the numpy arrays
    if flatten:
        X_train = X_train.reshape([X_train.shape[0], -1])
        X_val = X_val.reshape([X_val.shape[0], -1])
        X_test = X_test.reshape([X_test.shape[0], -1])
    return X_train, y_train, X_val, y_val, X_test, y_test


X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(flatten=True)

# Defining the network
network = []

# Adding layers
network.append(Dense(X_train.shape[1],100))
network.append(ReLU())
#network.append(Sigmoid())
network.append(Dense(100,200))
network.append(ReLU())
#network.append(Sigmoid())
network.append(Dense(200,300))
network.append(ReLU())
#network.append(Sigmoid())
network.append(Dense(300,400))
network.append(ReLU())
#network.append(Sigmoid())
network.append(Dense(400,10))


def forward(network, X):
    # Computes activations of all network layers by applying them sequentially.
    # Returns a list of activations for each layer. 
    
    activations = []
    input = X
    
    # Looping through each layer
    for layer in network:
        activations.append(layer.forward(input))
        # Updating input to last layer output
        input = activations[-1]
    
    assert len(activations) == len(network)
    return activations

def predict(network,X):
    
    # Computing network predictions. Returning indices of largest Logit probability
    logits = forward(network,X)[-1]
    return logits.argmax(axis=-1)


def train(network,X,y):
    
    # Get the layer activations
    # Firstly getting the list of layer activations
    layer_activations = forward(network,X) 
    # These are going to be the input for network, i.e layer_input[i] as input for network[i]
    layer_inputs = [X] + layer_activations 
    logits = layer_activations[-1]
    
    # Computing the loss and the initial gradient
    loss = softmax_crossentropy_with_logits(logits,y)
    loss_grad = grad_softmax_crossentropy_with_logits(logits,y)
    
    # Propagating gradients through the network
    # Reverse propogation as this is backpropogation
    
    for layer_index in range(len(network))[::-1]:
        layer = network[layer_index]
        
        loss_grad = layer.backward(layer_inputs[layer_index],loss_grad) #grad w.r.t. input, also weight updates
        
    return np.mean(loss)

from tqdm import trange
from IPython.display import clear_output

# Dividing the data into minibatches, updating the weights, which is known
# as minibatch stochastic gradient descent

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    
    assert len(inputs) == len(targets)
    
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in trange(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# Keeping the logs of accuracies
train_log = []
val_log = []

# Using 250 epochs. 
for epoch in range(250):
    for x_batch,y_batch in iterate_minibatches(X_train,y_train,batchsize=32,shuffle=True):
        train(network,x_batch,y_batch)
    
    train_log.append(np.mean(predict(network,X_train)==y_train))
    val_log.append(np.mean(predict(network,X_val)==y_val))
    
    clear_output()

    print("Epoch",epoch)
    print("Train accuracy:",train_log[-1])
    print("Validation accuracy:",val_log[-1])

    plt.plot(train_log,label='Train accuracy')
    plt.plot(val_log,label='Validation accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.show()



"""# Convolutional Neural Network

1. Using PyTorch
2. Using Keras

Compare these two in the report
"""

import torch
import torchvision
import torchvision.transforms as transforms

# The torchvision datasets are downloaded and normalised in the first section

import torch.nn as nn
import torch.nn.functional as F

# Defining the Convolutional Neural Network
# The class Net has 2 layers and uses ReLu as the activation function.

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Defining the channels of the CNN, modifying it to have 3 instead of 1
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):

        # Using ReLu as the activation function
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialising the CNN
net = Net()

import torch.optim as optim

# Our loss function is the Cross Entropy Loss.
# Stochastic Gradient Descent with momentum is used as the optimiser

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Tranining the CNN. 2 epochs are used, where this hyperparameter can be changed

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # Getting the inputs
        inputs, labels = data

        # Parameter gradients are zero
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Printing statistics
        running_loss += loss.item()
        # Looping through and printing minibatches
        if i % 2000 == 1999:   
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# Testing the CNN

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

outputs = net(images)

# Printing what the CNN predicted
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

# Testing Performance

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
      100*correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

"""# KERAS"""

import numpy as np

from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical   


# Loading in the data.
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Normalising the data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Testing the shape of the data
print("Shape of training data:")
print(X_train.shape)
print(y_train.shape)
print("Shape of test data:")
print(X_test.shape)
print(y_test.shape)

from keras.layers import Dense, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential 
from keras.optimizers import SGD


# Building the CNN using the built-in Keras functions. The CNN will have 2 layers
# and uses ReLu as an activation function, SGD with momentum as the optimiser function

model = Sequential()

# Adding the layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the channels and pixel arrays
model.add(Flatten())

# Testing different activation functions
model.add(Dense(256, activation='relu'))

# Also testing the model with Softmax.
model.add(Dense(10, activation='softmax'))

# Setting Stochastic gradient descent
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)

# Training the CNN

# Fitting the model using 15 epochs and a validation split of 0.2
history = model.fit(X_train, y_train, batch_size=32, epochs=15, verbose=2, validation_split=0.2)

# The plot function
def plotLosses(history):  
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


plotLosses(history)

# Evaluating the CNN

# Getting the scores of the model
score = model.evaluate(X_test, y_test, batch_size=128, verbose=0)

print(model.metrics_names)
print(score)

# The low accuracy might be due to overfitting. Therefore we can use
# regularisation to improve results.

# Regularisation

from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

# Setting the mode;
model = Sequential()

# Adding the layers, using ReLu as the activation function
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Adding a dropout layer here
model.add(Dropout(0.25))

# Flattening channels
model.add(Flatten())
model.add(Dense(256, activation='relu'))

# Another dropout layer added here
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Stochastic gradient descent with 0.9 momentum
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)

# Training the new CNN

history = model.fit(X_train, y_train, batch_size=32, epochs=15, verbose=2, validation_split=0.2)

# Plotting the performance
plotLosses(history)

score = model.evaluate(X_test, y_test, batch_size=128, verbose=0)

# Plotting scores
print(model.metrics_names)
print(score)

# Now using Batch Normalisation

from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

# Building the model, again
model = Sequential()

# 
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3)))

# Now we add batch normalisation.
model.add(BatchNormalization())
# Activation function is still ReLu
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Keeping the dropout layer
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
# Another batch normalization layer added here
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# SGD
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)

# Training the CNN
history = model.fit(X_train, y_train, batch_size=32, epochs=15, verbose=2, validation_split=0.2)
plotLosses(history)

# Evaluating the model

score = model.evaluate(X_test, y_test, batch_size=128, verbose=0)

# Plotting scores
print(model.metrics_names)
print(score)

# Using Data Augmentation ( ADAM )
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True)   # flip images horizontally

validation_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(X_train[:40000], y_train[:40000], batch_size=32)
validation_generator = validation_datagen.flow(X_train[40000:], y_train[40000:], batch_size=32)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3)))

# Batch normalization layer added here
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
# Batch normalization layer added here
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

adam = Adam(lr=0.0006, beta_1=0.9, beta_2=0.999, decay=0.0)

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

history = model.fit_generator(train_generator,    
                    validation_data=validation_generator,
                    validation_steps=len(X_train[40000:]) / 32,
                    steps_per_epoch=len(X_train[:40000]) / 32,
                    epochs=15,
                    verbose=2)

plotLosses(history)

score = model.evaluate(X_test, y_test, batch_size=128, verbose=0)
print(model.metrics_names)
print(score)



