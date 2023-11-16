import os
import sys
import csv
import time
import glob
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
import torchvision.models as models
import torch 
import torch.nn as nn
import torch.optim as optim
from utils import model_loader,PGD_attack_generator,SIP_data_loader
from autoattack import AutoAttack
sys.path.append("../")

# Description of all argument
parser = argparse.ArgumentParser()
parser.add_argument('-batchSize', type=int, default=1)
parser.add_argument('-data_split_path', required=False, default= '../data/formatted_train_test_split.csv',type=str)
parser.add_argument('-datasetPath', required=False, default= '../data/images/',type=str)
parser.add_argument('-model_weights_path',type=str)
parser.add_argument('-output_path',type=str)
parser.add_argument('-network', default= 'densenet',type=str,choices=["resnet","densenet","inception"])
parser.add_argument('-nClasses', default= 200,type=int)
parser.add_argument('-pretrained', action='store',nargs='?',const="IMAGENET1K_V1",default=None)
parser.add_argument('-epsilon', default=8,type=float)
parser.add_argument('-num_attack_iters',default=32,type=int)
parser.add_argument('-attack_step_size',default=1./255,type=float)
parser.add_argument('-attack_norm',default="L-INF",type=str)
parser.add_argument('-test_run_num', default=1,type=int)

args = parser.parse_args()
device = torch.device('cuda')

# Select the correct image size based on what model is being used
if args.network == "inception":
    image_size = 299
else:
    image_size = 224

model = model_loader.load_model(model_architecture=args.network, pretraining=args.pretrained, num_classes=args.nClasses,weights_path=args.model_weights_path)
top_1_results = 0
top_3_results = 0
top_5_results = 0
base_save_dir = args.output_path
os.makedirs(base_save_dir,exist_ok=True)
base_save_path = base_save_dir + f"/epsilon-{args.epsilon}_run-{args.test_run_num}"
model = model.to(device)
for param in model.parameters():
    param.requires_grad = False
model.eval()
attack_generator = AutoAttack(model, norm='Linf', eps=args.epsilon/255., version='standard',verbose=False)

# Set up our data loader
test_set_loader = SIP_data_loader.create_SIP_DataLoader(data_path=args.datasetPath, annotations_file=args.data_split_path, image_size=image_size, batch_size=args.batchSize,shuffle=True)

# Perform Adversarial Testing
for batch_idx, (data, cls) in enumerate(test_set_loader):
    ground_truth_labels = cls.numpy()
    data = data.to(device)
    cls = cls.to(device)
    adv_x = attack_generator.run_standard_evaluation(data,cls,bs=len(data))
    with torch.no_grad():
        outputs = model(adv_x)
        top_k_predictions = torch.topk(outputs,k=5,dim=-1)[1]
        top_k_predictions = top_k_predictions.detach().to("cpu").numpy()
        for ground_truth_label,top_predicted_labels in zip(ground_truth_labels,top_k_predictions):
            if ground_truth_label == top_predicted_labels[0]:
                top_1_results = top_1_results + 1
            if ground_truth_label in top_predicted_labels[:3]:
                top_3_results = top_3_results + 1
            if ground_truth_label in top_predicted_labels[:5]:
                top_5_results = top_5_results + 1
    del adv_x,outputs,top_k_predictions

print(f"Top-1 Accuracy: {top_1_results} | Top-3 Accuracy: {top_3_results} | Top-5 Accuracy: {top_5_results}")
with open ("_".join([base_save_path,"top-1.txt"]),"w") as f:
    f.write("%d" % top_1_results)
with open ("_".join([base_save_path,"top-3.txt"]),"w") as f:
    f.write("%d" % top_3_results)
with open ("_".join([base_save_path,"top-5.txt"]),"w") as f:
    f.write("%d" % top_5_results)
