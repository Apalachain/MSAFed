import torch
import torch.nn as nn
import numpy as np
from tqdm import trange
import copy
from numpy.linalg import norm
from collections import deque
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

class Memory(object):
    """
        Create the empty memory buffer
    """

    def __init__(self,num_sample,args,num_classes=2,mode=None):
        
        self.memory = {}
        self.num_sample = num_sample 
        self.args=args
        self.cache=self.num_sample*self.args.number_sample 
        if mode=="server":
            self.cache=self.num_sample*len(self.args.source) 
        self.num_classes=num_classes
        self.number_sample=self.args.number_sample
        self.size=self.cache*self.num_classes
        for i in range(self.num_classes):
            self.memory[i]=deque(maxlen=self.cache)
        
    def get_size(self):
        if len(self.memory[0])==0:
            return 0
        length=0
        for i in range(self.num_classes):
            length+=len(self.memory[i])
        return length 
    def get_class_size(self,class_label):
        return len(self.memory[class_label])

    

    def push(self,features,y):
        """
        for each class , impelement the principle of FIFO
        
        I intend to use dictionary to store the feature
        
        """

        batch=features.shape[0]
        labels=np.arange(self.num_classes)
        for point_id in range(batch):
            for label in labels:
                indices=(y[point_id]==label).nonzero(as_tuple=False)
                if self.number_sample<=len(indices):
                    sampled_indices=indices[torch.randperm(len(indices))[:self.number_sample]]
                else:
                    sampled_indices=indices 
                pixel_features=features[point_id,sampled_indices].squeeze()
                self.memory[label].extend(pixel_features)


    def _ini_test(self,features,class_label):
        if class_label in self.memory:
            self.memory[class_label].extend(features)


    def get_class_elements(self,class_label):
        """
        Get all elements for a specific class label as a torch tensor
        """
        if class_label in self.memory:
            elements=torch.stack(list(self.memory[class_label]))
            return elements.clone().detach()
        else:
            return torch.tensor([])

    def agg_func(self,groups,features):
        """
        aggeragate the pixel features to obtain the prototype
        """
        agg_proto=dict()
        for idx,group_idx in enumerate(groups):
            if group_idx in agg_proto:
                agg_proto[group_idx].append(features[idx])
            else:
                agg_proto[group_idx]=[]
                agg_proto[group_idx].append(features[idx])
        
        group_prototype=dict()

        for idx in range(len(agg_proto)):
            pro=sum(agg_proto[idx])/len(agg_proto[idx])
            group_prototype[idx]=pro
        return group_prototype

    def cluster(self,class_label,num_cluster):
        """
        cluster the pixel-wise feature to reduce the computation cost (group_prototype)
        """
        features=(self.get_class_elements(class_label)).cpu().numpy()
        kmeans=KMeans(n_clusters=num_cluster,n_init="auto",random_state=0)
        groups=kmeans.fit_predict(features)
        for i in range(num_cluster):
            print("the number of features in group {} is {}".format(i,np.sum(groups==i)))
        group_prototype=self.agg_func(groups,features)
        return group_prototype
    
    def store_prototype(self):
        class_labels=np.arange(self.args.num_class)
        prototypes=[]
        for class_label in class_labels:
            group_prototype=self.cluster(class_label,self.args.num_cluster)
            prototypes.append(group_prototype)
        return prototypes 

