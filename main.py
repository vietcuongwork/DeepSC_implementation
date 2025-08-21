import argparse
import json
import os
import random
import signal
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EurDataset, collate_data
from models.mutual_info import Mine
from models.transceiver import DeepSC
from utils import SNR_to_noise, train_step, val_step, train_mi, \
    plot_losses_from_checkpoints, clean_checkpoints, initNetParams, \
    SeqtoText, list_checkpoints, load_checkpoint

plt.ion()  # Turn on interactive mode

# Argument parser for configuring hyperparameters and paths
parser = argparse.ArgumentParser()
parser.add_argument('--vocab-file', default='vocab.json', type=str)
parser.add_argument('--checkpoint-path',
                    default='checkpoints/deepsc-AWGN',
                    type=str)
parser.add_argument('--channel', default='AWGN', type=str,
                    help='Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=80, type=int)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
stop_training = False  # Global variable to control training interruption


# Signal handler for stopping training gracefully
def signal_handler(sig, frame):
    global stop_training
    print("\nTraining interruption signal received. Saving progress...")
    stop_training = True


# Function to set random seed for reproducibility
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# Validation function
def validate(epoch, args, net, seq_to_text):
    test_eur = EurDataset('test')  # Load test dataset
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size,
                               num_workers=0, pin_memory=True,
                               collate_fn=collate_data)

    # # Print a sample batch from test_iterator for debugging
    # sample_batch = next(iter(test_iterator))  # Get first batch
    # print("Sample batch from test_iterator (Tensor format):")
    # print(sample_batch)  # Print the raw tensor

    # # Convert token IDs to text using sequence_to_text method
    # decoded_sentences = [
    #     seq_to_text.sequence_to_text(sent.cpu().numpy().tolist()) for sent in
    #     sample_batch]
    # print("\nSample batch (Decoded sentences):")
    # for i, sent in enumerate(decoded_sentences):
    #     print(f"Sentence {i + 1}: {sent}")

    # print_padded_sentences(test_iterator, seq_to_text, pad_idx)

    net.eval()
    pbar = tqdm(test_iterator)
    total = 0
    # Noise_std for TimeVaryingRician
    noise_std_options = np.arange(0.045, 0.316, 0.010)
    noise_std = np.random.choice(noise_std_options, size=1)
    with torch.no_grad():
        for sents in pbar:
            # print(f"Batch contains {sents.shape[0]} sentences")
            sents = sents.to(device)
            # loss, snr = val_step(net, sents, sents, 0.1, pad_idx, criterion,
            #                      args.channel, seq_to_text)
            # TimeVaryingRician
            loss, snr = val_step(net, sents, sents, 0.18, pad_idx, criterion,
                                 args.channel, seq_to_text)
            total += loss
            pbar.set_description(
                f'Epoch: {epoch + 1}; Type: VAL; Loss: {loss:.5f}')
    return total / len(test_iterator)


# Training function
def train(epoch, args, net, mi_net=None):
    global stop_training
    train_eur = EurDataset('train')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size,
                                num_workers=0, pin_memory=True,
                                collate_fn=collate_data)
    pbar = tqdm(train_iterator)
    # For TimeVaryingRician
    # noise_std_options = np.arange(0.045, 0.316, 0.010)
    epoch_loss = 0
    mi_bits_total = 0
    batch_count = 0
    snr_values = []

    for sents in pbar:
        if stop_training:
            return True, epoch_loss, mi_bits_total / batch_count if batch_count > 0 else 0, min(
                snr_values) if snr_values else 0, max(
                snr_values) if snr_values else 0, sum(snr_values) / len(
                snr_values) if snr_values else 0
        sents = sents.to(device)
        # noise_std = np.random.choice(noise_std_options, size=1).item()  # Scalar
        # For original Channel
        noise_std = float(
            np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))[0])
        if mi_net is not None:
            mi_loss, mi_bits = train_mi(net, mi_net, sents, 0.1, pad_idx,
                                        mi_opt, args.channel)
            loss_total, snr = train_step(net, sents, sents, 0.1, pad_idx,
                                         optimizer, criterion, args.channel,
                                         mi_net)
            epoch_loss += loss_total
            mi_bits_total += mi_bits
            batch_count += 1
            snr_values.append(snr)
            pbar.set_description(
                f'Epoch: {epoch + 1}; Type: Train; Loss: {loss_total:.5f}; MI Loss: {mi_loss:.5f}; MI (bits): {mi_bits:.5f}; SNR: {snr:.5f}')
        else:
            loss_total, snr = train_step(net, sents, sents, noise_std, pad_idx,
                                         optimizer, criterion, args.channel)
            epoch_loss += loss_total
            snr_values.append(snr)
            pbar.set_description(
                f'Epoch: {epoch + 1}; Type: Train; Loss: {loss_total:.5f}; SNR: {snr:.5f}; Noise Std: {noise_std:.5f}')

    snr_min = min(snr_values) if snr_values else 0
    snr_max = max(snr_values) if snr_values else 0
    snr_avg = sum(snr_values) / len(snr_values) if snr_values else 0

    avg_epoch_loss = epoch_loss / len(train_iterator)
    avg_mi_bits = mi_bits_total / batch_count if batch_count > 0 else 0
    return False, avg_epoch_loss, avg_mi_bits, snr_min, snr_max, snr_avg


if __name__ == '__main__':

    # count_sentences('data/train_data.pkl', 'data/test_data.pkl')
    # inspect_dataset('data/europarl/txt/en')
    # inspect_file('data/europarl/txt/en/ep-07-05-23-005-04.txt')
    # Define loss file path

    # Check PyTorch's CUDA availability
    print("PyTorch Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())
    print("CUDA Version:", torch.version.cuda)
    print("GPU Count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))

    signal.signal(signal.SIGINT, signal_handler)  # Bind Ctrl+C to stop training
    args = parser.parse_args()
    args.vocab_file = os.path.join('data',
                                   args.vocab_file)  # Simplified path joining
    loss_file = os.path.join(args.checkpoint_path, 'losses.json')

    # Print the selected channel
    print(f"Selected Channel: {args.channel}")

    # Load vocabulary file
    with open(args.vocab_file, 'rb') as f:
        vocab = json.load(f)
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]
    seq_to_text = SeqtoText(token_to_idx, end_idx)

    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab, num_vocab, num_vocab,
                    args.d_model, args.num_heads, args.dff, 0.1).to(device)
    mi_net = Mine().to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(deepsc.parameters(), lr=1e-4,
                                 betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)
    mi_opt = torch.optim.Adam(mi_net.parameters(), lr=0.001)

    initNetParams(deepsc)

    # List available checkpoints
    list_checkpoints(args.checkpoint_path)

    # Prompt user for action with input validation
    while True:
        action = input("Choose action: resume or start? ").strip().lower()
        if action in ['resume', 'start']:
            break
        print("Invalid input. Please enter 'resume' or 'start'.")

    start_epoch = 0
    if action == 'resume':
        checkpoint = load_checkpoint(args.checkpoint_path, mode='latest')
        if checkpoint and checkpoint['epoch'] < args.epochs:
            start_epoch = checkpoint['epoch']
            deepsc.load_state_dict(checkpoint['model_state_dict'])
            mi_net.load_state_dict(checkpoint['mi_net_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            mi_opt.load_state_dict(checkpoint['mi_opt_state_dict'])
            print(
                f"Resuming from epoch {start_epoch} with loss {checkpoint['loss']:.5f}")
        else:
            print(
                "Cannot resume: Training completed or no valid checkpoint. Switching to 'start'.")

    if action == 'start':
        checkpoint = load_checkpoint(args.checkpoint_path, mode='best')
        if checkpoint:
            deepsc.load_state_dict(checkpoint['model_state_dict'])
            best_loss = checkpoint['loss']
            print(f"Starting new phase with best model, loss {best_loss:.5f}")
        else:
            print("Starting from scratch: No best checkpoint found.")
        start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        start = time.time()
        # Training
        interrupted, epoch_train_loss, avg_mi_bits, snr_min, snr_max, snr_avg = train(
            epoch, args, deepsc)
        if interrupted:
            print(
                f"Training stopped at epoch {epoch + 1}. Saving checkpoint...")
            avg_loss = validate(epoch, args, deepsc, seq_to_text)
            checkpoint_path = os.path.join(args.checkpoint_path,
                                           f'checkpoint_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pth')
            os.makedirs(args.checkpoint_path, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': deepsc.state_dict(),
                'mi_net_state_dict': mi_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mi_opt_state_dict': mi_opt.state_dict(),
                'loss': avg_loss,
                'train_loss': epoch_train_loss,
                'mi_bits': avg_mi_bits,
                'snr_min': snr_min,
                'snr_max': snr_max,
                'snr_avg': snr_avg,
            }, checkpoint_path)
            print(
                f"Checkpoint saved at {checkpoint_path} with epoch {epoch + 1}, validation loss {avg_loss:.5f}, SNR Min: {snr_min:.2f}, Max: {snr_max:.2f}, Avg: {snr_avg:.2f}")
            break

        avg_loss = validate(epoch, args, deepsc, seq_to_text)
        checkpoint_path = os.path.join(args.checkpoint_path,
                                       f'checkpoint_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pth')
        os.makedirs(args.checkpoint_path, exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': deepsc.state_dict(),
            'mi_net_state_dict': mi_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'mi_opt_state_dict': mi_opt.state_dict(),
            'loss': avg_loss,
            'train_loss': epoch_train_loss,
            'mi_bits': avg_mi_bits,
            'snr_min': snr_min,
            'snr_max': snr_max,
            'snr_avg': snr_avg,
        }, checkpoint_path)
        print(
            f"Checkpoint saved at {checkpoint_path} with epoch {epoch + 1}, validation loss {avg_loss:.5f}, SNR Min: {snr_min:.2f}, Max: {snr_max:.2f}, Avg: {snr_avg:.2f}")

        print(f"GPU Utilization: {torch.cuda.utilization(0)}%")
        print(
            f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2} MB")

    # clean_checkpoints("checkpoints/deepsc-Rayleigh", keep_latest_n=5)

    print("Training finished.")
