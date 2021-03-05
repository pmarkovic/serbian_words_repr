import json
import numpy as np


class DataHandler:
    """
    DataHandler is the class that deals with preparing data for training.
    And saving parameters after training.
    """

    def __init__(self, 
                train_set_path, 
                word2ind_path, 
                noise_dist_path,
                is_sg, 
                window_size, 
                neg_samples,
                batch_size):

        self.train_set_path = train_set_path
        self.is_sg = is_sg
        self.word2ind = None
        self.voc_size = None
        self.noise_dist = None
        self.max_ws = window_size
        self.neg_samples = neg_samples
        self.bs = batch_size

        self._init_dictionaries(word2ind_path, noise_dist_path)


    def _init_dictionaries(self, word2ind_path, noise_dist_path):
        with open(word2ind_path, 'r') as json_file:
            self.word2ind = json.load(json_file)
        
        self.voc_size = len(self.word2ind)

        with open(noise_dist_path, 'r') as json_file:
            self.noise_dist = json.load(json_file)

    def get_voc_size(self):
        return self.voc_size
    
    def get_examples(self):
        """
        Method to obtain examples for training.
        Methods for both models are iterators that
        read line by line from the corpus and return a batch of examples.
        """
        if self.is_sg:
            return self.get_sg_example()
        
        return self.get_cbow_example()

    def get_sg_example(self):
        centar_indices = []
        context_indices = []
        neg_samples_indices = []

        # Need for sampling negative examples
        tokens = list(self.noise_dist.keys())
        tokens_prob = list(self.noise_dist.values())

        with open(self.train_set_path, 'r') as txt_file:
            for line in txt_file:
                sent = line.strip().split()

                # Fix center word
                for pos, word in enumerate(sent):
                    center_ind = self.word2ind[word]
                    ws = np.random.randint(2, min(self.max_ws+1, len(sent)))

                    # Look for words around the center word
                    for w in range(-ws, ws+1):
                        context_pos = pos + w 

                        # Checks for index not to go out of bound 
                        # and to be different from center word's index
                        if 0 <= context_pos < len(sent) and context_pos != pos:
                            samples_ind = [self.word2ind[sample] 
                                      for sample in np.random.choice(tokens, size=5, p=tokens_prob)]

                            centar_indices.append(center_ind)
                            context_indices.append(self.word2ind[sent[context_pos]])
                            neg_samples_indices.append(samples_ind)
                
                # If there are no enough examples, continue to the next line in the corpus
                if len(centar_indices) < self.bs:
                    continue
                
                # Return only batch size examples
                yield centar_indices[:self.bs], context_indices[:self.bs], neg_samples_indices[:self.bs]
                
                # Remove examples that are sent for training
                centar_indices = centar_indices[self.bs:]
                context_indices = context_indices[self.bs:]
                neg_samples_indices = neg_samples_indices[self.bs:]
    
    def get_cbow_example(self):
        centar_pos = []
        other_pos = []
        neg_samples_pos = []

        # Need for sampling negative examples
        tokens = list(self.noise_dist.keys())
        tokens_prob = list(self.noise_dist.values())

        with open(self.train_set_path, 'r') as txt_file:
            for line in txt_file:
                sent = line.strip().split()

                # Fix center word
                for pos, word in enumerate(sent):
                    center_ind = self.word2ind[word]
                    ws = np.random.randint(2, min(self.max_ws+1, len(sent)))
                    context_words = []

                    # Look for words around the center word
                    for w in range(-ws, ws+1):
                        context_pos = pos + w 

                        # Checks for index not to go out of bound 
                        # and to be different from center word's index
                        if 0 <= context_pos < len(sent) and context_pos != pos:
                            context_words.append(self.word2ind[sent[context_pos]])
                        
                    centar_pos.append(center_ind)
                    other_pos.append(context_words)

                    samples_ind = [self.word2ind[sample] 
                                      for sample in np.random.choice(tokens, size=5, p=tokens_prob)]
                    neg_samples_pos.append(samples_ind)
                
                # If there are no enough examples, continue to the next line in the corpus
                if len(centar_pos) < self.bs:
                    continue
                
                # Return only batch size examples
                yield centar_pos[:self.bs], other_pos[:self.bs], neg_samples_pos[:self.bs]

                # Remove examples that are sent for training
                centar_pos = centar_pos[self.bs:]
                other_pos = other_pos[self.bs:]
                neg_samples_pos = neg_samples_pos[self.bs:]

    def save_params(self, params, path):
        # Required in order to be able to convert in numpy array
        if params.requires_grad:
            params = params.detach()

        # Required in order to be able to convert in numpy array
        if params.is_cuda:
            params = params.cpu()

        with open(path, 'w') as txt_file:
            for ind, p in enumerate(params.numpy()):
                txt_file.write(f"{ind},{p}\n")
