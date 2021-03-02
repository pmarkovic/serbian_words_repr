import os
import json
import argparse
import numpy as np
import torch


class DataHandler:

    def __init__(self, 
                train_set_path, 
                word2ind_path, 
                noise_dist_path,
                is_sg, 
                ws, 
                neg_samples):

        self.train_set_path = train_set_path
        self.is_sg = is_sg
        self.word2ind = None
        self.voc_size = None
        self.noise_dist = None
        self.max_ws = ws 
        self.neg_samples = neg_samples

        self._init_dictionaries(word2ind_path, noise_dist_path)


    def _init_dictionaries(self, word2ind_path, noise_dist_path):
        with open(word2ind_path, 'r') as json_file:
            self.word2ind = json.load(json_file)
        
        self.voc_size = len(self.word2ind)

        with open(noise_dist_path, 'r') as json_file:
            self.noise_dist = json.load(json_file)

    def get_voc_size(self):
        return self.voc_size

    def get_one_hot_encoding(self, word_idx):
        x = torch.zeros(self.voc_size).float()
        x[word_idx] = 1.0

        return x
    
    def get_examples(self):
        if self.is_sg:
            return self.get_sg_example()
        
        return self.get_cbow_example()

    def get_sg_example(self):
        tokens = list(self.noise_dist.keys())
        tokens_prob = list(self.noise_dist.values())
        counter = 0

        with open(self.train_set_path, 'r') as txt_file:
            for line in txt_file:
                counter += 1
            
                sent = line.strip().split()
                for pos, word in enumerate(sent):
                    center_ind = self.word2ind[word]
                    ws = np.random.randint(2, min(self.max_ws+1, len(sent)))

                    for w in range(-ws, ws+1):
                        context_pos = pos + w 

                        if 0 <= context_pos < len(sent) and context_pos != pos:
                            samples_ind = [self.word2ind[sample] 
                                        for sample in np.random.choice(tokens, size=5, p=tokens_prob)]
                            yield center_ind, self.word2ind[sent[context_pos]], samples_ind
    
    def get_cbow_example(self):
        pass

    def save_params(self, params, path):
        with open(path, 'w') as txt_file:
            for ind, p in enumerate(params.numpy()):
                txt_file.write(f"{ind},{p}\n")


