#pip3 install torch torchvision torchaudio <- this is already installed on colab but for other coding envirormnts we would use pip to install these

#in positron instal as follows in the console to get pytorch:
#%pip install torch torchvision torchaudio

# For reading data
import gzip
import numpy as np
import struct
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# For visualizing
import plotly.express as px

# For model building
import torch
import torch.nn as nn
import torch.nn.functional as F

#as we use neurel nets we cant just use panda data frames.
#we also need to use mini batches to minimize memory use (read in mini batches of data)

#we have to make a data object unique to the data were using becasse data will have different unique shapes depenidng on the data we use
#were going to have to create a specififc data class for every set of data (this may or may not make neural networs worth using becuase they will take A LOt of effort)
#this is the main reason we cant just use pandas

class FashionMNISTDataset(Dataset):
    def __init__(self, gz_image_file, gz_label_file, transform=None):
        with gzip.open(gz_image_file, 'rb') as f:
            magic, num_images = struct.unpack(">II", f.read(8))
            rows, cols = struct.unpack(">II", f.read(8))
            self.images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)

        with gzip.open(gz_label_file, 'rb') as f:
            magic, num_labels = struct.unpack(">II", f.read(8))
            self.labels = np.frombuffer(f.read(), dtype=np.uint8)

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        #this will reshape the image to (1, 28, 28) before returning it back
        image = image.reshape(1, 28, 28)
        # this will make the image a float32 tensor, yayyyyyy (finally!)
        image = torch.tensor(image, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label


train_data = FashionMNISTDataset("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz")  # Pass both image and label files
test_data = FashionMNISTDataset("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")   # Pass both image and label files

# Create data feed pipelines for modeling, this will take our custom data object and sample them in small batches
train_dataloader = DataLoader(train_data, batch_size=64) #this will get them ready for our neural network, think of data loaders as the pipeline that feed the data into our machine (the neural network)
test_dataloader = DataLoader(test_data, batch_size=64)

import matplotlib.pyplot as plt

labels_map={
    0: 'T-shirt',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle Boot',
}

idx=1  #as we go through the different index observations we should get back the image and it should tell us what it is (see both image and the value)

#want get image labels instead of numbers (like from the in class example)

image, label = train_data.__getitem__(idx)

#grab the label from the labels_map created above

text_label = labels_map[label]

#print out the answer::

print(f"This is image is labeled a {text_label}")
px.imshow(image.reshape(28, 28))

#got rid of theses because these will give us numbers intead of text (matching)
#print(f"This image is labeled a {train_data.__getitem__(idx)[1]}")
#px.imshow(train_data.__getitem__(idx)[0].reshape(28, 28))


#I think I got it right for the most part! Yayyyy

#lets build our first network

class FirstNet(nn.Module):  #inherheriting from out neural network module (nn) from above
    def __init__(self):  #we create our initialization function

      # We define the components of our model here
      super(FirstNet, self).__init__() #were goinging to initalize with all of the values that come with the nn.Module object (this is where we get all of the initalization stuff already built in by that objecy)

      # Function to flatten our image
      self.flatten = nn.Flatten()  #then we create a flatten object, using pytorch's flatten object



      # Create the sequence of our network
      self.linear_relu_model = nn.Sequential(    #now create a linear model, this will be a sequentialled model from our nn library with a single in it (10 perceptrons)
            # Add a linear output layer w/ 10 perceptrons
            nn.LazyLinear(10),   #the 10 perceptrons look at all the inputs and do their best guessing and combine their info, this will be the layer that gets us our 10 classes. Each neuron represents a different class. Here we wil learn about which pixels matter more for which class of number (0 through 9). Each lable gets its own neuron in this output layer. See how likely each of these outcomes are
        )

#next we define our forward function (i)

    def forward(self, x): #x are our inputs, so this is where the data will flow into this forward network
      # We construct the sequencing of our model here
      x = self.flatten(x) #x comes in and the we update x to flatten it back out in 748 pixels (note this after we made it 1x28x28 above)
      # Pass flattened images through our sequence
      output = self.linear_relu_model(x) #then feed the xs through our linear model here, and our model is those 10 perceptrons we talked about above

      # Return the evaluations of our ten
      #   classes as a 10-dimensional vector
      return output #we then pass those 10 different values as output

# Create an instance of our model (our class that we created)
model = FirstNet()


#now we start training our neural network (this is going to take a while)

# Define some training parameters
learning_rate = 1e-3
batch_size = 64
epochs = 17 #every epoch means we pass every observation to our model 1 times, so if we do 20, then our model will have seen the data 20 times

# Define our loss function
#   This one works for multiclass problems
loss_fn = nn.CrossEntropyLoss() #loss function is for how we grade our performance , cross entropy loss is good for multiclass problems. use this all as ameasure of performance

#now we need to build an optimizer
# Build our optimizer with the parameters from
#   the model we defined, and the learning rate
#   that we picked
optimizer = torch.optim.SGD(model.parameters(), #stotastic geadidi desecent model. Take model parameters and learning  rate . optimizer chooses how to advance our model
     lr=learning_rate)

#now we prepare to train the model

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset) #check how big our datat set is
    # Set the model to training mode
    # important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train() #set our model into trainig mode, when testing we stop updating our model (stop back poprgation we dont want model to be updating)
    # Loop over batches via the dataloader
    for batch, (X, y) in enumerate(dataloader): #loop over our batches in the data loader
        # Compute prediction and loss
        pred = model(X) #make predictions based on the current model
        loss = loss_fn(pred, y) #calcualte loss in our model based on those predictions

        # Backpropagation and looking for improved gradients
        loss.backward() #now go backwards, look for best way to change our model
        optimizer.step() #step our model in that direction
        # Zeroing out the gradient (otherwise they are summed)
        #   in preparation for next round
        optimizer.zero_grad() #zero out our optimizier

        # Print progress update every few loops (every 10 batches, print out what has happened so far)
        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]") #will go through this loop over and over again with each batch, epoch (our training loop is a single epocoh )
        
#now also need a test loop
#note that we dont have an optimizer here (because were not updating the model, instead we set it to evaluation mode)
def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode
    # important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0 #looks at how many observations we get correct

    # Evaluating the model with torch.no_grad() ensures
    # that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations
    # and memory usage for tensors with requires_grad=True
    with torch.no_grad(): #not looking for gradient
        for X, y in dataloader:
            pred = model(X) #go through and make predictions
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # Printing some output after a testing round
    test_loss /= num_batches #this will give our average output (loss divided by the number of batches )
    correct /= size #how many we got right as a % of the number of observations
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

#now we train our model with a for loop

# Need to repeat the training process for each epoch.
#   In each epoch, the model will eventually see EVERY
#   observations in the data
for t in range(epochs): #trains once for every epoch,
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer) #run our trainig loop through the entire data set
    test_loop(test_dataloader, model, loss_fn) #then run 1 test loop, then it will move on to the next epch
print("Done!") #print out each poc

#careful because if you run this again it will continue to update the model! (it will start at where you stopped last time (from the last epoch you stopped on))



#make more predictions

# Decide if we are loading for predictions or more training
model.eval()
# - or -
# model.train()

# Make predictions
pred = model(test_data.__getitem__(1)[0]).argmax()
truth = test_data.__getitem__(1)[1]

#now switch theses so that they give the text label (instead of a number)

pred_text = labels_map[pred.item()]
truth_text = labels_map[truth]

#print(f"This image is predicted to be a {pred}, and is labeled as {truth}") #this one gives numbers

print(f"This image is predicted to be a {pred_text}, and is labeled as {truth_text}")

#our model so far is still like 40 years out of date lol

#saving a model, importing  and continuing to update it from there 
#this matters because energy consuptions matters! Very wasteful from an environemntal and cost perspective to do create these models over and over
#thus its important to save our models so we can use them later!

# Save our model for later, so we can train more or make predictions

EPOCH = epochs #stores our epoch
# We use the .pt file extension by convention for saving
#    pytorch models
PATH = "model.pt" #create a psth and then save the moedl

# The save function creates a binary storing all our data for us
torch.save({ #pass in a dictionary of all the stuff you want to save
            'epoch': EPOCH,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, PATH) #path is the file were going to save this info into 

            #save weights, biases, as well as epochs and optimozers into the file

#should be able to find saved file over in the files now, (will be model.pt in this case)


#now to load back in our model in our next session we do the following: 

# Specify our path
PATH = "model.pt"

# Create a new "blank" model to load our information into (has no updated weights and biases)
model = FirstNet()

# Recreate our optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) #(has no updated weights and biases)

# Load back all of our data from the file (puts all that stuff back in mmeory, this means we dont have to train the model everytime we use it)
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
EPOCH = checkpoint['epoch']