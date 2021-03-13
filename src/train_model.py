import time
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from azureml.core import Run

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
    parser.add_argument("--lr", default=0.025,
                        help="Learning rate for optimizer (default=0.025).")
    parser.add_argument("--seed", default=892,
                        help="Seed for random numbers generator (default=892).")
    parser.add_argument("--wind_size", default=3, 
                        help="Max window size for surrounding context words (default=3).")
    parser.add_argument("--neg_sample", default=5,
                        help="Number of negative samples to pick (default=5).")
    parser.add_argument("--batch_size", default=128,
                        help="Size of batches during training (default=128).")

    args = parser.parse_args()

    return args


def train(args):
    """
    Function for training models.
    """

    embed_dim = args.embed_dim
    num_epochs = args.epochs
    init_lr = args.lr

    # For monitoring on Azure
    run = Run.get_context()

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
                               args.is_sg, args.wind_size, args.neg_sample, args.batch_size)

    voc_size = data_handler.get_voc_size()
    print(f"Voc size: {voc_size}")

    if args.is_sg:
        model = SkipGram(voc_size, embed_dim)
    else:
        model = CBOW(voc_size, embed_dim)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=init_lr)

    loss_per_epoch = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} started...")
        epoch_loss = 0
        cycle_loss = 0

        # Update lr by decreasing it linearly for every epoch
        lr = init_lr * (1.0 - (1.0 * epoch) / num_epochs)
        print(f"Lr: {lr}")
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for batch_ind, batch in enumerate(data_handler.get_examples()):
            # Create one hot encoding for every tokens (center, context, neg samples) and examples

            vi = F.one_hot(torch.tensor(batch[0]), num_classes=voc_size).float().to(device)
        
            if args.is_sg:
                vo = F.one_hot(torch.tensor(batch[1]), num_classes=voc_size).float().to(device)
            else:
                # Had to do padding due to different number of context tokes per example
                one_hot = [F.pad(F.one_hot(torch.tensor(e), num_classes=voc_size), 
                                            (0, 0, 0, 2*args.wind_size-len(e))) for e in batch[1]]
                vo = torch.stack(one_hot).float().to(device)

            neg_samples = F.one_hot(torch.tensor(batch[2]), num_classes=voc_size).float().to(device)

            # Calculate loss
            curr_loss = model(vi, vo, neg_samples)
            epoch_loss += curr_loss
            cycle_loss += curr_loss
            curr_loss.backward()

            optimizer.step()

            optimizer.zero_grad()

            if batch_ind % 100 == 0:
                print(f"After {batch_ind} batches, loss: {cycle_loss / 100}")
                cycle_loss = 0

        print(f"Loss after {epoch+1} epoch: {epoch_loss / batch_ind}")
        loss_per_epoch.append(epoch_loss / batch_ind)

        run.log('loss', epoch_loss / batch_ind)
        print(f"Total number of examples: {batch_ind * args.batch_size}")
    
    params = model.get_trained_parameters()
    print("Saving model parameters...")
    data_handler.save_params(params, args.params_path)
    print("Training is finished...")

    print(f"Epoch losses: {' '.join(list(map(str, loss_per_epoch)))}")


if __name__ == "__main__":
    start_tme = time.time()
    args = arg_parser()

    train(args)

    print(f"Full training time: {round(time.time() - start_tme, 2)}s.")

