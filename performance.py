#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Author: Huiqiang Xie
@File: performance.py
@Time: 2021/4/1 11:48
@Modified: 2025/03/25 - Process one SNR at a time and save results incrementally
"""

import argparse
import gc
import json
import os
from sklearn.preprocessing import normalize as sk_normalize
import numpy as np
import torch
from torch.nn.functional import normalize
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from utils import BleuScore, SNR_to_noise, greedy_decode, SeqtoText, \
    save_evaluation_scores, load_checkpoint

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='train_data.pkl', type=str)
parser.add_argument('--vocab-file', default='vocab.json', type=str)
parser.add_argument('--checkpoint-path',
                    default='checkpoints/deepsc-Rayleigh',
                    type=str)
parser.add_argument('--channel', default='Rayleigh', type=str)
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--bert-config-path',
                    default='bert/cased_L-12_H-768_A-12/bert_config.json',
                    type=str)
parser.add_argument('--bert-checkpoint-path',
                    default='bert/cased_L-12_H-768_A-12/bert_model.ckpt',
                    type=str)
parser.add_argument('--bert-dict-path',
                    default='bert/cased_L-12_H-768_A-12/vocab.txt', type=str)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Similarity:
    def __init__(self, batch_size=4):
        """Initialize BERT model and tokenizer for sentence similarity computation"""
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.model = BertModel.from_pretrained('bert-base-cased')
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size

    def prepare_sentences(self, sentences, max_length=32):
        """Tokenize and prepare sentences for BERT processing"""
        encoded = self.tokenizer(sentences, padding=True, truncation=True,
                                 max_length=max_length, return_tensors='pt')
        return encoded['input_ids'].to(self.device), encoded[
            'attention_mask'].to(self.device)

    def get_sentence_embeddings(self, sentences):
        """Convert sentences to embeddings using BERT in batches"""
        all_embeddings = []
        for i in range(0, len(sentences), self.batch_size):
            batch_sentences = sentences[i:i + self.batch_size]
            input_ids, attention_mask = self.prepare_sentences(batch_sentences)
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     output_hidden_states=True)
                hidden_states = outputs.hidden_states[
                    -2]  # Second-to-last layer
                # Sum all token embeddings without masking (original approach)
                sum_embeddings = torch.sum(hidden_states, dim=1)
                all_embeddings.append(sum_embeddings.cpu())
                del outputs, hidden_states
                torch.cuda.empty_cache()
        return torch.cat(all_embeddings, dim=0).to(self.device)

    def compute_similarity(self, real, predicted):
        """Compute similarity scores between real and predicted sentences"""
        if not real or not predicted:
            return []
        chunk_size = 100
        all_similarities = []
        total_chunks = (len(real) + chunk_size - 1) // chunk_size
        with tqdm(total=total_chunks, desc="Computing Similarity") as pbar:
            for i in range(0, len(real), chunk_size):
                torch.cuda.empty_cache()
                gc.collect()
                chunk_real = real[i:i + chunk_size]
                chunk_predicted = predicted[i:i + chunk_size]
                real_embeddings = self.get_sentence_embeddings(chunk_real)
                pred_embeddings = self.get_sentence_embeddings(chunk_predicted)
                # Convert to NumPy for max normalization
                real_embeddings_np = real_embeddings.cpu().numpy()
                pred_embeddings_np = pred_embeddings.cpu().numpy()
                # Apply max normalization (feature-wise)
                real_embeddings_norm = sk_normalize(real_embeddings_np,
                                                    norm='max', axis=0)
                pred_embeddings_norm = sk_normalize(pred_embeddings_np,
                                                    norm='max', axis=0)
                # Compute cosine similarity manually
                dot = np.sum(real_embeddings_norm * pred_embeddings_norm,
                             axis=1)
                a = np.sqrt(np.sum(real_embeddings_norm ** 2, axis=1))
                b = np.sqrt(np.sum(pred_embeddings_norm ** 2, axis=1))
                similarities = dot / (a * b)
                all_similarities.extend(similarities.tolist())
                del real_embeddings, pred_embeddings, real_embeddings_np, pred_embeddings_np
                torch.cuda.empty_cache()
                pbar.update(1)
        return all_similarities


def memory_status():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        print(
            f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
        print(
            f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1e9:.2f}GB")


def performance(args, SNR, net):
    print("\nInitializing evaluation metrics...")
    similarity = Similarity(batch_size=4)
    bleu_score_1gram = BleuScore(1, 0, 0, 0)

    print("Loading test dataset...")
    test_eur = EurDataset('test')
    StoT = SeqtoText(token_to_idx, end_idx)

    final_bleu = []
    final_sim = []

    # Define the maximum number of samples to process per SNR per epoch
    MAX_SAMPLES = -1  # Set to -1 to process the entire dataset, or a positive integer to limit samples

    # Determine the number of samples to process per epoch
    if MAX_SAMPLES == -1:
        total_samples_per_epoch = len(test_eur)
        print(
            f"Processing full test dataset with {len(test_eur)} samples per SNR per epoch.")
    else:
        total_samples_per_epoch = min(MAX_SAMPLES, len(test_eur))
        print(
            f"Processing limited to {total_samples_per_epoch} samples per SNR per epoch.")

    print(
        f"Starting evaluation for {len(SNR)} SNR levels, {args.epochs} epochs each...")
    net.eval()
    with torch.no_grad():
        for snr in SNR:
            print(f"\nProcessing SNR: {snr} dB")
            noise_std = SNR_to_noise(snr)
            score_bleu_snr = []
            score_sim_snr = []

            for epoch in range(args.epochs):
                print(f"\nEpoch {epoch + 1}/{args.epochs} for SNR {snr} dB")
                word = []  # Transmitted words for this epoch
                target_word = []  # Received words for this epoch

                # Reload DataLoader for each epoch
                test_iterator = DataLoader(test_eur, batch_size=args.batch_size,
                                           num_workers=0, pin_memory=True,
                                           collate_fn=collate_data)

                samples_processed = 0  # Track the number of samples processed

                # Progress bar to monitor sample processing
                with tqdm(total=total_samples_per_epoch,
                          desc=f"SNR {snr} dB - Epoch {epoch + 1}") as pbar:
                    for batch_idx, sents in enumerate(test_iterator):
                        if samples_processed >= total_samples_per_epoch:
                            break  # Stop once we've processed the desired number of samples

                        sents = sents.to(device)
                        target = sents
                        out, snr_value = greedy_decode(net, sents, noise_std,
                                                       args.MAX_LENGTH, pad_idx,
                                                       start_idx,
                                                       args.channel, device)
                        sentences = out.cpu().numpy().tolist()
                        result_string = list(
                            map(StoT.sequence_to_text, sentences))
                        word.extend(result_string)
                        target_sent = target.cpu().numpy().tolist()
                        result_string = list(
                            map(StoT.sequence_to_text, target_sent))
                        target_word.extend(result_string)

                        # Update samples processed
                        batch_size = sents.size(0)
                        samples_processed += batch_size
                        pbar.update(batch_size)

                        # Trim excess samples if we exceed the limit
                        if samples_processed > total_samples_per_epoch:
                            excess = samples_processed - total_samples_per_epoch
                            word = word[:-excess]
                            target_word = target_word[:-excess]
                            samples_processed = total_samples_per_epoch
                            break

                # Print the total number of samples processed
                print(f"Total samples processed: {len(word)}")

                print(
                    "\nComputing BLEU and similarity scores for this epoch...")
                torch.cuda.empty_cache()
                gc.collect()
                print("Memory status before computation:")
                memory_status()

                # Compute scores
                bleu = bleu_score_1gram.compute_blue_score(word, target_word)
                bleu_mean = np.mean(bleu)
                score_bleu_snr.append(bleu_mean)

                sim = similarity.compute_similarity(word, target_word)
                sim_mean = np.mean(sim)
                score_sim_snr.append(sim_mean)

                print(f"BLEU score: {bleu_mean:.4f}")
                print(f"Similarity score: {sim_mean:.4f}")
                print("Memory status after computation:")
                memory_status()

                # Clear memory
                word = []
                target_word = []
                torch.cuda.empty_cache()
                gc.collect()

            # Average scores across epochs for this SNR
            snr_bleu = np.mean(score_bleu_snr)
            snr_sim = np.mean(score_sim_snr)
            final_bleu.append(snr_bleu)
            final_sim.append(snr_sim)

            print(f"\nSNR {snr} dB Summary:")
            print(f"Average BLEU score across epochs: {snr_bleu:.4f}")
            print(f"Average Similarity score across epochs: {snr_sim:.4f}")

            # Save results for this SNR
            save_evaluation_scores(args, [snr], [snr_bleu], [snr_sim], 'DeepSC',
                                   1)

    print("\nFinal Results:")
    print("SNR levels:", SNR)
    print("Average BLEU scores:", final_bleu)
    print("Average Similarity scores:", final_sim)

    # Save final aggregated results
    save_evaluation_scores(args, SNR, final_bleu, final_sim, 'DeepSC', 1)
    return final_bleu, final_sim


if __name__ == '__main__':
    args = parser.parse_args()
    # SNR = [0, 3, 6, 9, 12, 15, 18]
    SNR = [18]
    args.vocab_file = './data/' + args.vocab_file
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab, num_vocab, num_vocab,
                    args.d_model, args.num_heads, args.dff, 0.1).to(device)

    checkpoint = load_checkpoint(args.checkpoint_path, mode='best')
    if checkpoint:
        deepsc.load_state_dict(checkpoint['model_state_dict'])
        best_loss = checkpoint['loss']
        print(f"Loaded best checkpoint with loss {best_loss:.5f}")
    else:
        print("No best checkpoint found.")

    bleu_score, similarity_score = performance(args, SNR, deepsc)
    print(bleu_score)
    print(similarity_score)
