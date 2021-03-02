import time
import argparse
import numpy as np

import torch
import torch.optim as optim

from models import SkipGram, CBOW
from data_handler import DataHandler


def arg_parser():
    """ takes user input """

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default="data/train_set.txt",
                        help="Path to the training data (default=data/train_set.txt).")
    parser.add_argument("--w2i_path", default="data/word2ind.json",
                        help="Path to the word2ind dict (default=data/word2ind.json).")
    parser.add_argument("--nd_path", default="data/noise_dist.json",
                        help="Path to the noise dist dict (default=data/noise_dist.json).")
    parser.add_argument("--params_path", default="data/params.txt", 
                        help="Path where to store trained parameters/embeddings (default=data/params.txt).")
    parser.add_argument("--is_sg", default=False, action="store_true",
                        help="Flag to indicate which model to use (default=False).")
    parser.add_argument("--embed_dim", default=300,
                        help="Dimension of embeddings (default=300).")
    parser.add_argument("--epochs", default=5,
                        help="Number of training epochs (default=5).")
    parser.add_argument("--lr", default=0.01,
                        help="Learning rate for optimizer (default=0.01).")
    parser.add_argument("--seed", default=892,
                        help="Seed for random numbers generator (default=892).")
    parser.add_argument("--wind_size", default=5, 
                        help="Max window size for surrounding context words (default=5).")
    parser.add_argument("--neg_sample", default=5,
                        help="Number of negative samples to pick (default=5).")
    parser.add_argument("--max_example", default=100,
                        help="For testing on smaller number of examples (default=100).")

    args = parser.parse_args()

    return args


def train(args):
    embed_dim = args.embed_dim
    num_epochs = args.epochs
    lr = args.lr

    # For the reproducibility purpose
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("Model:")
    print(f"Is SG: {args.is_sg}")
    print(f"Embed dim: {embed_dim}")

    data_handler = DataHandler(args.data_path, args.w2i_path, args.nd_path,
                               args.is_sg, args.wind_size, args.neg_sample)

    if args.is_sg:
        model = SkipGram(data_handler.get_voc_size(), embed_dim)
    else:
        model = CBOW(data_handler.get_voc_size(), embed_dim)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} started...")
        loss = 0
        num_examples = 0

        for vi_ind, vo_ind, neg_samples_ind in data_handler.get_examples():
            vi = data_handler.get_one_hot_encoding(vi_ind).to(device)

            if args.is_sg:
                vo = data_handler.get_one_hot_encoding(vo_ind).to(device)
            else:
                vo = torch.stack(list(map(data_handler.get_one_hot_encoding, 
                                            vo_ind))).to(device)
            neg_samples = torch.stack(list(map(data_handler.get_one_hot_encoding, 
                                                neg_samples_ind))).to(device)
            num_examples += 1

            curr_loss = model(vi, vo, neg_samples)
            loss += curr_loss
            curr_loss.backward()

            optimizer.step()

            optimizer.zero_grad()

            if num_examples == int(args.max_example):
                break

        print(f"Loss after {epoch+1} epoch: {loss / num_examples}")
    
    params = model.get_trained_parameters()
    print("Saving model parameters...")
    data_handler.save_params(params, args.params_path)


if __name__ == "__main__":
    start_tme = time.time()
    args = arg_parser()

    train(args)

    print(f"Full training time: {round(time.time() - start_tme, 2)}s.")

