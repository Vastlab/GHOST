import torch
import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import sys
import GHOST 
import pdb
import argparse
import os
import cv2
import glob
###
# In this example, logits and feature from the penultimate layer were extracted from a networks and stored in a local folder "Extractions"
# This experimental setup was based on ImageNet2012 as known training data. 
# Output scores are saved into a "GHOST_Runs" folder

# Excamples of file names expected in the Extractions folder: 'mae_H_train_logits.npy' 'mae_H_train_FV.npy' 'mae_H_val_logits.npy' 'mae_H_val_FV.npy' 'mae_H_iNaturalist_logits.npy' 'mae_H_iNaturalist_FV.npy'
# Files were saved as numpy arrays in the following format:
# [[Ground_Truth, logit_0, logit_1, ... logit_999],
# [Ground_Truth, logit_0, logit_1, ... logit_999],
# [Ground_Truth, logit_0, logit_1, ... logit_999]]

#FVs were saved in the following format
#[[Ground_Truth, FV_Dim_0, FV_Dim_1, ... FV_Dim_999],
# [Ground_Truth, FV_Dim_0, FV_Dim_1, ... FV_Dim_999],
# [Ground_Truth, FV_Dim_0, FV_Dim_1, ... FV_Dim_999]]
#
# Where the zeroth dimension/axis corresponds to each sample


extractions_folder = "/scratch/rrabinow/GHOST/Extractions"

architectures = [] #Fancy way to determine what architectures are available in Extractions folder
ext_files = glob.glob(os.path.join(extractions_folder, "*"))
for path in ext_files:
    if '.npy' not in path or 'val_logits' not in path:
        continue
    filename = path.split("/")[-1]
    architectures.append(filename.replace("_val_logits.npy", ""))
    

parser = argparse.ArgumentParser()
parser.add_argument("--arch", type=str, required=True, choices=architectures, help="Underlying architecture")
parser.add_argument("--sys", type=str, required=True, choices=list(GHOST.choices.keys()), help="Normalization to use")#Choices come from GHOST library


args = parser.parse_args()

print("Start")




debug = False #Used for development only, speeds up fitting process
arch = args.arch
top_save_dir = 'GHOST_Runs/'
if not os.path.exists(top_save_dir):
    os.mkdir(top_save_dir)

run_folder = top_save_dir +arch+"_"+args.sys+"/" #This is where files will be saved

   
if not os.path.exists(run_folder):
    os.mkdir(run_folder)



print("Loading files")

training_path = os.path.join(extractions_folder,arch+"_train_logits.npy")
training_FV_path = os.path.join(extractions_folder,arch+"_train_FV.npy")


training_data = torch.from_numpy(np.load(training_path))
training_FV_data = torch.from_numpy(np.load(training_FV_path))

#Ensure data is in the same order between FV and logit files!
assert torch.all(training_data[:,0] == training_FV_data[:,0]) == True

print("Filtering data")

#Separate GT from logit data
training_gt = training_data[:,0]

#Separate logits from GT
training_logits = training_data[:,1:]

#Separate FV from GT
training_FV = training_FV_data[:,1:]

#Filter the training samples so we only use samples where max_logit == GT
filtered_train_mask = training_gt == torch.max(training_logits, dim=1).indices
filtered_train_logits = training_logits[filtered_train_mask]
filtered_train_FV = training_FV[filtered_train_mask]
filtered_train_gt = training_gt[filtered_train_mask]

if debug == True:
    sampled_logits = []
    sampled_FVs = []
    sampled_gt = []
    for class_id in torch.unique(filtered_train_gt):
        selected_logits = filtered_train_logits[filtered_train_gt == class_id]
        
        sampled_logits.append(selected_logits[:min(10, selected_logits.shape[0])])
        sampled_FVs.append(filtered_train_FV[filtered_train_gt == class_id][:min(10, selected_logits.shape[0])])
        sampled_gt.append(filtered_train_gt[filtered_train_gt == class_id][:min(10, selected_logits.shape[0])])
        
    filtered_train_logits = torch.cat(sampled_logits, dim=0)
    filtered_train_FV = torch.cat(sampled_FVs, dim=0)
    filtered_train_gt = torch.cat(sampled_gt, dim=0)

#################
#Fitting/Training
#################

GHOST_class = GHOST.choices[args.sys]
GHOST_model =  GHOST_class(filtered_train_logits, filtered_train_FV)
if args.sys == "GHOST":
    with open(run_folder+'Gaus_models.pkl', 'wb') as f:
        pickle.dump(GHOST_model.Gaus_dict, f)
    
print("Saved model")    

##########
#Inference
##########
plots = []
#Datasets to process NOTE:Assumes naming follows arch+"_"+dataset+"_logits.npy"
datasets = ['val', 'iNaturalist'] 
for dataset in datasets:
    print("dataset is set to "+ dataset)
    
    test_path = os.path.join(extractions_folder,arch+"_"+dataset+"_logits.npy")
    test_FV_path = os.path.join(extractions_folder,arch+"_"+dataset+"_FV.npy")
    
    
    test_data = torch.from_numpy(np.load(test_path))
    test_FV_data = torch.from_numpy(np.load(test_FV_path))
    

    test_gt = test_data[:,0] #Separate out GT
    
    if not 'val' in dataset: #If we're using an OOD dataset, force GT to be -1
        print(f'{dataset} is being treated as OOD')
        test_gt = (test_gt * 0) - 1
    
    test_logits = test_data[:,1:]
    test_FVs = test_FV_data[:,1:]
    test_preds = torch.max(test_logits, dim=1).indices
    
    
    #probs = GHOST_model.ReScore(test_logits, test_FVs, to_save={'dataset':dataset, 'path':run_folder})
    probs = GHOST_model.ReScore(test_logits, test_FVs)
    max_index_test = torch.max(test_logits, dim=1).indices

    
    #Save gt, pred, prob
    test_preds = torch.cat((test_gt.view(-1,1), max_index_test.view(-1,1), probs.view(-1,1)), dim=1)
        
    np.save(run_folder+"GHOST_"+dataset+"_preds.npy", arr=test_preds.numpy())

sys.exit()
    

