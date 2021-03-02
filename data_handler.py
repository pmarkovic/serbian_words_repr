import os
import json
import time
import argparse
import numpy as np


NOISE_DIST_PATH = os.path.join(os.getcwd(), "data", "noise_dist.json")


def arg_parser():
    """ takes user input """

    parser = argparse.ArgumentParser()

    parser.add_argument("--examples_path", default="data/examples.json", 
                        help="Path to the json file where to store training examples (default=data/examples.json).")
    parser.add_argument("--data_path", default="data/train_set.txt",
                        help="Path to the file where training data are stored (default=data/train_set.txt).")
    parser.add_argument("--seed", default=892,
                        help="Seed for random numbers generator (default=892).")
    parser.add_argument("--wind_size", default=5, 
                        help="Max window size for surrounding context words (default=5).")
    parser.add_argument("--neg_sample", default=5,
                        help="Number of negative samples to pick (default=5).")

    args = parser.parse_args()

    return args


class DataHandler:

    def __init__(self, 
                train_set_path, 
                word2ind_path, 
                noise_dist_path,
                is_sg=False, 
                ws=5, 
                neg_samples=5):

        self.train_set_path = train_set_path
        self.is_sg = is_sg
        self.word2ind = None
        self.noise_dist = None
        self.max_ws = ws 
        self.neg_samples = neg_samples

        self._init_dictionaries(word2ind_path, noise_dist_path)


    def _init_dictionaries(self, word2ind_path, noise_dist_path):
        with open(word2ind_path, 'r') as json_file:
            self.word2ind = json.load(json_file)

        with open(noise_dist_path, 'r') as json_file:
            self.noise_dist = json.load(json_file)

    
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

