import linecache
import numpy as np


class DataHandler:
    """
    DataHandler is the class that deals with preparing data for training.
    And saving parameters after training.
    """

    def __init__(self, 
                examples_path,
                voc_size,
                n_examples,
                is_sg,
                batch_size):

        self.examples_path = examples_path
        self.is_sg = is_sg
        self.voc_size = voc_size
        self.n_examples = n_examples
        self.bs = batch_size

    def get_examples(self):
        """
        Method to obtain examples for training.
        Random examples are chosen and then returned in batches.
        """

        centar_pos = []
        other_pos = []
        neg_samples_pos = []
        
        print("Choose indices")
        example_indices = np.random.choice(self.voc_size, size=self.n_examples, replace=False)

        print("Collecting examples...")
        for ind in example_indices:
            line = linecache.getline(self.examples_path, ind)
            example = line.strip().split(';')
            
            centar_pos.append(int(example[0]))

            if self.is_sg:
                other_pos.append(int(example[1]))
            else:
                other_pos.append(list(map(int, example[1].split(','))))

            neg_samples_pos.append(list(map(int, example[2].split(','))))

        print("Examples ready!")
        for i in range(0, len(centar_pos), self.bs):
            end = min(i+self.bs, len(centar_pos))

            yield centar_pos[i:end], other_pos[i:end], neg_samples_pos[i:end]
