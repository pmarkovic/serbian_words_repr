import json
import numpy as np

import torch
import torch.nn.functional as F

from models import SkipGram, CBOW



def get_cbow_example(bs, wind):
    centar_pos = []
    other_pos = []
    neg_samples_pos = []

    with open("./data/word2ind.json", 'r') as json_file:
        word2ind = json.load(json_file)

    with open("./data/train_set.txt", 'r') as txt_file:
        for line in txt_file:
            sent = line.strip().split()

            for pos, word in enumerate(sent):
                center_ind = word2ind[word]
                ws = np.random.randint(2, min(wind+1, len(sent)))
                context_words = []

                for w in range(-ws, ws+1):
                    context_pos = pos + w 

                    if 0 <= context_pos < len(sent) and context_pos != pos:
                        context_words.append(word2ind[sent[context_pos]])
                    
                centar_pos.append(center_ind)
                other_pos.append(context_words)
                neg_samples_pos.append(list(np.random.randint(100, 200, 4)))
            
            if len(centar_pos) < bs:
                continue
                
            yield centar_pos[:bs], other_pos[:bs], neg_samples_pos[:bs]
            centar_pos = centar_pos[bs:]
            other_pos = other_pos[bs:]
            neg_samples_pos = neg_samples_pos[bs:]

def get_sg_example(bs, wind):
    centar_pos = []
    other_pos = []
    neg_samples_pos = []

    with open("./data/word2ind.json", 'r') as json_file:
        word2ind = json.load(json_file)

    with open("./data/train_set.txt", 'r') as txt_file:
        for line in txt_file:
            sent = line.strip().split()

            for pos, word in enumerate(sent):
                center_ind = word2ind[word]
                ws = np.random.randint(2, min(wind+1, len(sent)))

                for w in range(-ws, ws+1):
                    context_pos = pos + w 

                    if 0 <= context_pos < len(sent) and context_pos != pos:
                        samples_ind = np.random.randint(100, 200, 4)

                        centar_pos.append(center_ind)
                        other_pos.append(word2ind[sent[context_pos]])
                        neg_samples_pos.append(list(samples_ind))
            
            if len(centar_pos) < bs:
                continue
                
            yield centar_pos[:bs], other_pos[:bs], neg_samples_pos[:bs]
            centar_pos = centar_pos[bs:]
            other_pos = other_pos[bs:]
            neg_samples_pos = neg_samples_pos[bs:]



def train():
    embed_dim = 5
    num_epochs = 1

    # For the reproducibility purpose
    torch.manual_seed(23)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(23)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("Model:")
    print(f"Is SG: {True}")
    print(f"Embed dim: {embed_dim}")

    if False:
        model = SkipGram(76431, embed_dim)
    else:
        model = CBOW(76431, embed_dim)
    model.to(device)

    for batch_ind, batch in enumerate(get_cbow_example(10, 2)):
        print(batch[1])
        break

        vi = F.one_hot(torch.tensor(batch[0]), num_classes=76431).float().to(device)
        vo = F.one_hot(torch.tensor(batch[1]), num_classes=76431).float().to(device)
        neg_samples = F.one_hot(torch.tensor(batch[2]), num_classes=76431).float().to(device)

        curr_loss = model(vi, vo, neg_samples)
        
        print(curr_loss)
        break


if __name__ == "__main__":
    train()

