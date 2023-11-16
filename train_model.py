import os
import sys
import csv
import time
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
from utils import model_loader,PGD_attack_generator,SIP_data_loader
sys.path.append("../")

def save_results(log,fieldnames,output_path):
    with open(os.path.join(output_path,"log.csv"),"w") as f:
        writer = csv.writer(f)
        limit = len(log[fieldnames[0]])
        for i in range(limit):
            writer.writerow([log[x][i] for x in fieldnames])
    return

# Description of all argument
parser = argparse.ArgumentParser()
parser.add_argument('-batchSize', type=int, default=32)
parser.add_argument('-nEpochs', type=int, default=50)
parser.add_argument('-data_split_path',type=str)
parser.add_argument('-datasetPath',type=str)
parser.add_argument('-outputPath',type=str)
parser.add_argument('-network', default= 'densenet',type=str,choices=["resnet","densenet","inception"])
parser.add_argument('-nClasses', default= 200,type=int)
parser.add_argument('-pretrained', action='store',nargs='?',const="IMAGENET1K_V1",default=None)
parser.add_argument('-learning_rate', default=0.1,type=float)
parser.add_argument('-epsilon', default=8.,type=float)
parser.add_argument('-num_attack_iters',default=32,type=int)
parser.add_argument('-attack_step_size',default=1.,type=float)
parser.add_argument('-use_mask',action="store_true")
parser.add_argument('-invert_mask',action="store_true")
parser.add_argument('-train_val_split',default=0.1)
parser.add_argument('-unannotated_split',default=0.0,type=float)
parser.add_argument('-salience_format',default=None,choices=[None,"salience_maps","bounding_boxes"])

args = parser.parse_args()
device = torch.device('cuda')

# Load the model
model = model_loader.load_model(model_architecture=args.network, pretraining=args.pretrained, num_classes=args.nClasses)
model = model.to(device)
epsilon = args.epsilon/255.
attack_step_size = args.attack_step_size/255.
attack_generator = PGD_attack_generator.PGD_Attack_Generator(model,epsilon,args.num_attack_iters,attack_step_size)

# Select the correct image size based on what model is being used
if args.network == "inception":
    image_size = 299
else:
    image_size = 224

# Set up our data loader
train_loader,val_loader = CUB_data_loader.create_SIP_DataLoader(data_path=args.datasetPath, annotations_file=args.data_split_path, image_size=image_size, batch_size=args.batchSize,salience_form=args.salience_format,val_split=args.train_val_split,unannotatied_split=args.unannotated_split,default_salience_map_path="zeros_img.png",shuffle=True,pin_memory=True)

# Create destination folder
output_path = args.outputPath
os.makedirs(output_path,exist_ok=True)
best_model_path = os.path.join(output_path,"best_model.pth")

# Description of hyperparameters
solver = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=1e-6, momentum=0.9)
lr_sched = optim.lr_scheduler.StepLR(solver, step_size=10, gamma=0.75)

# Set up our cross-entropy loss and our salience loss
criterion = nn.CrossEntropyLoss()

# File for logging the training process
log = {'epoch':["epoch"],'train_loss':["train_loss"],'train_accuracy':["train_accuracy"],'test_loss':["test_loss"],'test_accuracy':["test_accuracy"]}
fieldnames=["epoch","train_loss","train_accuracy","test_loss","test_accuracy"]

#####################################################################################
#
############### Training of the model and logging ###################################
#
#####################################################################################
bestAccuracy = 0
bestEpoch=0
for epoch in range(args.nEpochs):
    log['epoch'].append(epoch+1)
    start = time.perf_counter()

    # Training Phase
    # Set model to the correct mode and set up variables to keep track during training
    model.train()
    total_loss_over_epoch = 0.
    total_correct_predictions_over_epoch = 0
    total_number_of_samples_over_epoch = 0
    number_of_batches = 0

    # Loop through the data
    with torch.set_grad_enabled(True):
        for batch_idx, (data, cls, hmap) in enumerate(train_loader):
            # Ensure data is in the correct format
            batch_size = len(data)
            data = data.to(device)
            cls = cls.to(device)

            # Perform Adversarial Attacks
            if args.use_mask:
                hmap = hmap.to(device)
                if args.invert_mask:
                    mask_obj = hmap
                else:
                    mask_obj = torch.ones_like(hmap)-hmap
                mask_obj = torch.where(mask_obj >= 0.5,1.0,0.0)
                data = attack_generator(data,cls,mask=mask_obj)
            else:
                data = attack_generator(data,cls)

            # Get model predictions and accuracy
            outputs = model(data)
            predicted_classes = torch.max(outputs,dim=1)[1]
            num_correct_predictions = torch.sum((predicted_classes == cls).int()).item()
            total_correct_predictions_over_epoch += num_correct_predictions
            total_number_of_samples_over_epoch += batch_size

            # Calculate our various losses
            loss = criterion(outputs, cls)
            total_loss_over_epoch += loss.item()

            # Backpropagate our losses
            solver.zero_grad()
            loss.backward()
            solver.step()
            number_of_batches += 1
            del loss,data,hmap,outputs,predicted_classes,cls

    # Log training results
    accuracy = total_correct_predictions_over_epoch/total_number_of_samples_over_epoch
    mean_loss_over_epoch = total_loss_over_epoch/number_of_batches
    print('Epoch: ', epoch, 'Train loss: ',mean_loss_over_epoch, 'Accuracy: ', accuracy)
    log['train_loss'].append(mean_loss_over_epoch)
    log['train_accuracy'].append(accuracy)

    # Testing Phase
    # Switch model to evaluation mode and set up variables to keep track of results during testing
    model.eval()
    total_loss_over_epoch = 0.
    total_correct_predictions_over_epoch = 0
    total_number_of_samples_over_epoch = 0
    number_of_batches = 0

    with torch.set_grad_enabled(False):
        for batch_idx, (data, cls, hmap) in enumerate(val_loader):
            batch_size = len(data)
            data = data.to(device)
            cls = cls.to(device)

            # Get test predictions and accuracies
            outputs = model(data)
            predicted_classes = torch.max(outputs,dim=1)[1]
            num_correct_predictions = torch.sum((predicted_classes == cls).int()).item()
            total_correct_predictions_over_epoch += num_correct_predictions
            total_number_of_samples_over_epoch += batch_size

            # Calculate the test loss
            class_loss = criterion(outputs, cls)
            total_loss_over_epoch = total_loss_over_epoch + class_loss.item()
            number_of_batches += 1

    # Log test results
    accuracy = total_correct_predictions_over_epoch/total_number_of_samples_over_epoch
    mean_loss_over_epoch = total_loss_over_epoch/number_of_batches
    print('Epoch: ', epoch, 'Test loss:', mean_loss_over_epoch, 'Accuracy: ', accuracy)
    log['test_loss'].append(mean_loss_over_epoch)
    log['test_accuracy'].append(accuracy)
    lr_sched.step()
    if (accuracy >= bestAccuracy):
        bestAccuracy = accuracy
        bestEpoch = epoch
        states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': solver.state_dict(),
                }
        torch.save(states, best_model_path)
    save_results(log,fieldnames,output_path)
    end = time.perf_counter()
    print(f"Ellapsed Time {end-start}")

# Save the log of our model during training
save_results(log,fieldnames,output_path)

# Plotting of train and test loss
plt.figure()
plt.xlabel('Epoch Count')
plt.ylabel('Loss')
plt.plot(np.arange(0, args.nEpochs), log["train_loss"][1:], color='r')
plt.plot(np.arange(0, args.nEpochs), log["test_loss"][1:], 'b')
plt.legend(('Train Loss', 'Validation Loss'), loc='upper right')
plt.savefig(os.path.join(output_path,'model_Loss.jpg'))

# Plotting of train and test accuracy
plt.figure()
plt.xlabel('Epoch Count')
plt.ylabel('Accuracy')
plt.plot(np.arange(0, args.nEpochs), log["train_accuracy"][1:], color='r')
plt.plot(np.arange(0, args.nEpochs), log["test_accuracy"][1:], 'b')
plt.legend(('Train Accuracy', 'Validation Accuracy'), loc='upper right')
plt.savefig(os.path.join(output_path,'model_Accuracy.jpg'))
