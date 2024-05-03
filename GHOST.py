import torch
import numpy as np
import scipy
import sys
from tqdm import tqdm
import pdb
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import scipy
import os
#from vast.DistributionModels import weibull

import multiprocessing as mp
n_cpu = int(os.cpu_count()*0.8)



 
        
def single_gpd_fit(name, data):
    return name, scipy.stats.genpareto.fit(data)

class GHOST_Base:
    #Data must be a 2-D tensor containing the logits of correctly classified samples (where dim-0 is samples, dim-1 is vectors)
    #Optionally, rescale can be a 1-D tensor of scalar values associated with each sample in data
    def __init__(self, data, FVs): 
        print("GHOST init")
        self.GHOST_params = {}
        

        print("Starting norm")
        mask_for_gpd_fit = torch.randperm(data.shape[0])#[:50000]
        data = self.norm(data[mask_for_gpd_fit], FVs[mask_for_gpd_fit])

        
        
        print("Start GPD fit")
        
        self.gpd_mp_fit(data)
        

    def gpd_mp_fit(self, data):
        max_values, gt = torch.max(data, dim=1)
        
        pool = mp.Pool(processes=n_cpu)
        jobs = []
        for class_id in np.unique(gt.numpy()):
            class_data = data[gt == class_id][:,class_id]
            job = pool.apply_async(func=single_gpd_fit, args=(class_id, class_data))
            jobs.append(job)

        for job in jobs: 
            class_id, gpd_param = job.get()
            shape,loc,scale = gpd_param
            self.GHOST_params[class_id] = (shape, loc, scale)


        pool.close()
        pool.join()
        return
    
    #Data must be a 2-D tensor containing the logits of all test time samples
    #Returns 1-D tensor containing probabilities that a sample belongs to the predicted known class
    def ReScore(self, data, FVs):

        normd_data = self.norm(data, FVs)
        
        max_data, pred = torch.max(data, dim=1)
        
        rescored_data = torch.zeros(data.shape[0])
        for class_id in range(data.shape[1]):
            class_data = normd_data[pred == class_id][:,class_id]
            
            shape, loc, scale = self.GHOST_params[class_id]
            
            rescored_class_data = torch.from_numpy(scipy.stats.genpareto.cdf(class_data, shape, loc=loc, scale=scale))
            
            rescored_data[pred == class_id] = rescored_class_data.float()
        
        return rescored_data
    
    def norm(self, data, FVs):
        pass


class GHOST(GHOST_Base):
    def __init__(self, data, FVs):
        self.Gaus_dict = self.Gaus_gen(data, FVs)
        GHOST_Base.__init__(self, data, FVs)
        
        return
        
        
    def norm(self, logits, FVs):
        pred = torch.max(logits, dim=1).indices
        normalized_logits = torch.zeros(logits.shape)

        print("Iterating classes for Guassian scoring")
        for c in tqdm(self.Gaus_dict.keys()):
            class_mask = pred == c
            mean_vector, std_vector = self.Gaus_dict[c]
            FV_Z_Score = (FVs[class_mask]-mean_vector)/std_vector
            FV_Z_Score = torch.abs(FV_Z_Score) #Distance is distance
            diff_score = torch.sum(FV_Z_Score, dim=1)

            normalized_logits[class_mask,c] = logits[class_mask,c]/diff_score
            if normalized_logits.isnan().any() or normalized_logits.isinf().any():
                print("finite issue encountered")
                pdb.set_trace()

        return normalized_logits

    def Gaus_gen(self, logits, FV): #ASSUME DATA IS CORRECTLY PREDICTED
        preds = torch.max(logits, dim=1).indices
        classes = torch.unique(preds).long().tolist()

        class_models = {}
        print("Generating gaussian models")
        for c in tqdm(classes):
            select_class_FVs = FV[preds == c]

            mean = torch.mean(select_class_FVs, dim=0)
            std = torch.std(select_class_FVs, dim=0)
            
            std[std == 0] = torch.inf #This basically nullifies feature dimensions with constant values (eg. 0 standard deviation)

            class_models[c] = (mean, std)

        return class_models
    
    def normalize_features(self, logits, FV): #Normalize features based on max logit predicted class.
        preds = torch.max(logits, dim=1).indices
        classes = torch.unique(preds).long().tolist()
        
        normalized_features = torch.zeros(FV.shape)
        for c in classes:
            mean_vector, std_vector = self.Gaus_dict[c]
            select_class_FVs = FV[preds == c]
            z_scores = torch.abs((select_class_FVs - mean_vector)/std_vector)
            normalized_features[preds==c] = FV[preds==c]/z_scores
            
        return normalized_features
    
    def normalize_features_wrt_class(self, class_num, FV): #Normalize features assuming they belong to a specific class
        normalized_features = torch.zeros(FV.shape)
        mean_vector, std_vector = self.Gaus_dict[class_num]
        z_scores = torch.abs((FV - mean_vector)/std_vector)
        normalized_features = FV/z_scores

        return normalized_features

choices = {
        "GHOST":GHOST,
    }

if __name__ == '__main__':
    print("")


    