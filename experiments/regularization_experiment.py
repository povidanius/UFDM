from locale import normalize
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.models import inception_v3, resnet18
from torch import optim
import sys
import os
from collections import OrderedDict
import time
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from sklearn.metrics import *

#torch.manual_seed(31337)


data_path='/home/tank/Downloads/melanoma/joined'

sys.path.insert(0, "../")
from kac_independence_measure import KacIndependenceMeasure
from torch.nn.functional import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


REGULARIZER = 0
LOSS = 1

kim = KacIndependenceMeasure(2, 2, lr=0.001, input_projection_dim = 0, weight_decay=0.1, device=device) #0.007
kim1 = KacIndependenceMeasure(128-20, 20, lr=0.001, input_projection_dim = 0, weight_decay=0.1, device=device) #0.007


train_transform = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.RandomHorizontalFlip(),
                                      #transforms.RandomVerticalFlip(), 
                                      #transforms.CenterCrop(150),                                      
                                      #transforms.RandomRotation(180),                                 
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

test_transform = transforms.Compose([transforms.Resize((224,224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

full_dataset = ImageFolder(data_path, transform=train_transform)


n_data = len(full_dataset)
num_train = int(0.5*n_data)
training_dataset, testing_dataset = torch.utils.data.random_split(full_dataset, [num_train, n_data - num_train])
len_train= len(training_dataset)
len_test= len(testing_dataset)


print("Train: {}, test: {}".format(len_train, len_test))
batch_size = 128 # 128

train_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset= testing_dataset, shuffle=False)


model = resnet18(pretrained=True) #, aux_logits=False)

model.fc = nn.Sequential(
    #nn.BatchNorm1d(512),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.BatchNorm1d(128),
    nn.Linear(128, 32),
    nn.ReLU(),
    nn.BatchNorm1d(32),
    nn.Linear(32, 2)
)

# intermediate activations

#model.fc[2].register_forward_hook(get_activation('bottleneck'))


model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr = 0.0002, weight_decay=0.00001)
#optimizer1 = torch.optim.AdamW(params=model.parameters() + kim.trainable_parameters, lr = 0.0002, weight_decay=0.00001)

train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []


dep_history = []


if len(sys.argv) < 2:
    print(sys.argv)
    breakpoint()
    print("Usage {} 1/0".format(sys.argv[0]))
    sys.exit(0)

if sys.argv[1] == "0":
    use_regularization = False
else:
    use_regularization = True


reg_alpha = 0.2 #0.1
kacim_normalization = True


number_of_epoch = 2
#if use_regularization:
#    number_of_epoch = 2*number_of_epoch
    

global_iteration = 0
for epoch in range(number_of_epoch):
    
    train_correct = 0
    test_correct = 0
    train_iter_loss = 0.0
    test_iter_loss = 0.0
    train_iteration = 0
    test_iteration = 0
    num_train = 0

    model.train()
    iteration = 0
    #train_loader_copy = train_loader.copy()
    for data,label in train_loader:
        
        global_iteration += 1

        optimizer.zero_grad()
        data = data.to(device)
        label = label.to(device)        
        pred = model(data)     

        #breakpoint()
        #bottleneck = activation['bottleneck'].squeeze()
        #bottleneck1 = bottleneck[:,:128-20]
        #bottleneck2 = bottleneck[:,128-20:]

        y = torch.nn.functional.one_hot(label).float()


        if use_regularization:
            reg0 = kim.forward(pred.clone().detach().to(device), y.clone().detach().to(device), update=True, normalize=kacim_normalization)
            reg = kim.forward(pred, y, update=False, normalize=kacim_normalization)
            dep_history.append(reg.detach().cpu().numpy())
            writer.add_scalar("Dep_min/train", reg, global_iteration)
            #writer.add_scalar("Loss/train", loss, global_iteration)
            #loss = 0.0*loss_fn(pred, label) 
            loss = (1.0 - reg_alpha) * loss_fn(pred, label)  - reg_alpha * reg # loss -> min.., dep -> max
            #loss = -1.0*reg        
            print("Loss iteration: epoch {}, iteration {}, loss {}, reg {}, reg0 {} ".format(epoch, iteration, loss, reg, reg0))
            writer.add_scalar("LossReg/train", loss, global_iteration)
        else:
            loss = loss_fn(pred, label) 

        loss.backward()
        optimizer.step()


        
        train_iter_loss += loss.item()
        train_iteration += 1
        
        _, predicted = torch.max(pred, 1)
        train_correct += (predicted == label).sum()
        num_train += batch_size
        iteration = iteration + 1

     
    if train_iteration > 0:    
        train_loss.append(train_iter_loss/train_iteration)
        train_accuracy.append(100*float(train_correct)/num_train)
    else:
        continue
    #train_accuracy.append(100*float(train_correct)/len_train)
    
    if False:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            }, format("chest_checkpoint_{}.pt".format(epoch)))

    print ('Epoch {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f}'.format(epoch+1, number_of_epoch, train_loss[-1], train_accuracy[-1]))

#plt.plot(dep_history)
#plt.show()
#timestr = time.strftime("%Y%m%d-%H%M%S")
#plt.savefig('./chest_{}.png'.format(timestr))

    corrected = 0
    y_test_true = []
    y_test_pred = []
    preds = []
    ys = []
    model.eval()
    for data, label in test_loader:
        data = data.to(device)
        label = label.to(device)
        
        pred = model(data)
        y = torch.nn.functional.one_hot(label).float()


        _, predicted = torch.max(pred, 1)
        

        y_test_true.extend(label.clone().detach().cpu().numpy())
        y_test_pred.extend(predicted.clone().detach().cpu().numpy())
        #preds.extend(pred.clone().detach().to(device))
        #ys.extend(y.clone().detach().to(device))
        batch_correct = (predicted == label).sum()
        corrected += batch_correct
    
    #reg0 = kim.forward(preds.clone().detach().to(device), ys.clone().detach().to(device), update=False, normalize=kacim_normalization)
    #print("Testing: ".format(reg0))
    
    accuracy = 100 * float(corrected)/ len_test
    cm = confusion_matrix(y_test_true, y_test_pred, normalize='true') 
    #f1 = f1_score(y_test_true, y_test_pred)

    
    print(f'Test accuracy is {accuracy :.3f}, {cm[0,0] :.3f} { cm[0,1] :.3f} { cm[1,0] :.3f} { cm[1,1]:.3f}')
    print("Regularization: {}".format(use_regularization))
    writer.add_scalar("Acc/test", accuracy, global_iteration)

writer.close()

with open("./regularization_experiment_{}_{}.txt".format(use_regularization, reg_alpha),"a") as f:
    f.write("{},{},{},{} \n".format(cm[0,0], cm[0,1], cm[1,0], cm[1,1]))
    

