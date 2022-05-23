import csv
import os

# Command for Marijn:
import sys
#sys.path.append('/Users/marijnvandermeer/opt/anaconda3/envs/env_pytorch/lib/python3.6/site-packages')


from torch import empty
from linear import Linear
from relu import ReLU
from leakyrelu import LeakyReLU
from generate_dataset import generate_set
from sequential import Sequential
from gd import GD
from tanh import Tanh
from sgd import SGD
from mse import MSE
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import torch
torch.set_grad_enabled(False)


# ------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------
def data_split(data, batch_size):
    batch = [data[i:i + batch_size] for i in range(0, len(data), batch_size)] 
    return batch

def data_normalize(data):
    normalized = (data - data.mean())/data.std()    
    return normalized

# Parameters of plots:
SMALL_SIZE = 8
MEDIUM_SIZE = 16
BIGGER_SIZE = 22

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# ------------------------------------------------------------------
# Control the randomness
# ------------------------------------------------------------------
torch.manual_seed(0)

N = 1000
center = (0.5, 0.5)
r = 1 / (math.sqrt(2 * math.pi))

print('Generating test and training set')
training_data, training_labels = generate_set(N, center, r)
test_data, test_labels = generate_set(N, center, r)

print('Setting up model')
# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
print('Model: linear(2,25), ReLu(), linear(25,25), tanh(), linear(25,25), tanh(), linear(25,25), tanh(), linear(25,1)')
m1 = Linear(2, 25)
m2 = Linear(25, 25)
m3 = Linear(25, 25)
m4 = Linear(25, 25)
m5 = Linear(25, 1)

# Choose here between ReLU and Leaky ReLU:
r = ReLU()
#r = LeakyReLU(0.1)
tanh = Tanh()

# Sequential composition of model:
s = Sequential([m1, r,  m2, tanh, m4, tanh, m3, tanh, m5])
print('Parameters of model:\n{}\n'.format(s.param()))
# Parameters:
batch_size = 4
num_epochs = 25
lr = 1e-4
gd = GD(s.param(), lr)
mse = MSE()

# Split data:
training_batches = data_split(data_normalize(training_data), batch_size)
tlabel_batches = data_split(training_labels, batch_size)

test_batches = data_split(data_normalize(test_data), batch_size)
testlabel_batches = data_split(test_labels, batch_size)

# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------
avg_loss = torch.empty(len(training_batches)*num_epochs)
avg_loss_test = torch.empty(len(training_batches)*num_epochs)
m = 0
print('Start training!')
for j in range(num_epochs):
    print('# Epoch:{}/{}'.format(j, num_epochs))
    # ------------------------------------------------------------------
    # forward pass (using Mini Batch gradient descent)
    # ------------------------------------------------------------------
    for i in range(len(training_batches)):
        pred = s.forward(*tuple(training_batches[i]))
        inp = pred + tuple(tlabel_batches[i])  
        x = mse.forward(*inp)

        grad_loss = mse.backward()
        
        avg_loss_batch = torch.empty(len(x))
        
        for k in range(len(x)):
            avg_loss_batch[k] = x[k][0]
        
        # Evaluate on test set: 
        pred = s.forward(*tuple(test_batches[i]))
        inp_test = pred + tuple(testlabel_batches[i])  
        x = mse.forward(*inp_test)
        
        avg_loss_batch_test = torch.empty(len(x))
        for k in range(len(x)):
            avg_loss_batch_test[k] = x[k][0]        
        
        # ------------------------------------------------------------------
        # backward pass
        # ------------------------------------------------------------------
        output = s.backward(*grad_loss)

        # ------------------------------------------------------------------
        # Gradient step
        # ------------------------------------------------------------------
        gd.step()
        gd.zero_grad() 
        
        avg_loss[m] = avg_loss_batch.mean()
        avg_loss_test[m] = avg_loss_batch_test.mean()
        m+=1
        
print('Done!')
print('Final loss on training set: \n{}\n'.format(avg_loss[-1]))
print('Final loss on test set: \n{}\n'.format(avg_loss_test[-1]))


# Save loss: 

filename = 'training_loss.csv'
path_csv = "losses_csv/" + filename
os.makedirs(os.path.dirname(path_csv), exist_ok=True)
with open(path_csv, mode='w') as training_file:
    training_writer = csv.writer(training_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(avg_loss)):
        training_writer.writerow([avg_loss[i]])

filename = 'test_loss.csv'
path_csv = "losses_csv/" + filename
os.makedirs(os.path.dirname(path_csv), exist_ok=True)
with open(path_csv, mode='w') as test_file:
    test_writer = csv.writer(test_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(avg_loss_test)):
        test_writer.writerow([avg_loss_test[i]])

# Plot loss: 
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
axs[0].set_title('MSE loss evolution')
axs[0].set_xlabel('Num iterations')
axs[0].set_ylabel('Average batch loss')
axs[0].plot(avg_loss, label = 'training loss')
axs[0].plot(avg_loss_test, label = 'test loss')
axs[0].legend()

# Plot loss: 
axs[1].set_title('MSE for last 200 iterations')
axs[1].set_xlabel('Num iterations')
axs[1].set_ylabel('Average batch loss')
axs[1].plot(avg_loss[-200:], label = 'training loss')
axs[1].plot(avg_loss_test[-200:], label = 'test loss')
axs[1].legend()

# Save plot
filename = "Loss_project_2.png"
path = "plots/" + filename
os.makedirs(os.path.dirname(path), exist_ok=True)
fig.savefig(path, bbox_inches='tight')
print("plot saved as {} in plots folder!".format(filename))        

plt.show()