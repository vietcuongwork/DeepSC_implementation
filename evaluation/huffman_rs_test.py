#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File: huffman_rs_test.py
@Description: Tests Huffman + RS coding with 64-QAM, using sentence similarity calculation over a Rayleigh channel.
"""

import json
import random
import time
from torch.utils.data import Subset
import numpy as np
from dataset import EurDataset, collate_data
import huffman
from reedsolo import RSCodec
from utils import SNR_to_noise, SeqtoText, Channels
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import argparse
import os
from datetime import datetime
import pandas as pd
from transformers import BertModel, BertTokenizer
from sklearn.preprocessing import normalize as sk_normalize

# Device setup
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Vocabulary loading
VOCAB_FILE = './data/vocab.json'
vocab = json.load(open(VOCAB_FILE, 'rb'))
token_to_idx = vocab['token_to_idx']
idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
end_idx = token_to_idx["<END>"]
StoT = SeqtoText(token_to_idx, end_idx)


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
                chunk_real = real[i:i + chunk_size]
                chunk_predicted = predicted[i:i + chunk_size]
                real_embeddings = self.get_sentence_embeddings(chunk_real)
                pred_embeddings = self.get_sentence_embeddings(chunk_predicted)
                real_embeddings_np = real_embeddings.cpu().numpy()
                pred_embeddings_np = pred_embeddings.cpu().numpy()
                real_embeddings_norm = sk_normalize(real_embeddings_np,
                                                    norm='max', axis=0)
                pred_embeddings_norm = sk_normalize(pred_embeddings_np,
                                                    norm='max', axis=0)
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


class HuffmanCoding:
    def __init__(self):
        self.codebook = None
        self.reverse_codebook = None
        self.original_length = 0

    def compress(self, sentences):
        text = ' '.join(sentences)
        freq = {char: text.count(char) for char in set(text)}
        self.codebook = huffman.codebook(freq.items())
        self.reverse_codebook = {v: k for k, v in self.codebook.items()}
        encoded = ''.join(self.codebook[char] for char in text)
        self.original_length = len(encoded)
        return encoded

    def segment_compress(self, sentence, max_bits):
        compressed = self.compress([sentence])
        segments = [compressed[i:i + max_bits] for i in
                    range(0, len(compressed), max_bits)]
        if len(segments[-1]) < max_bits:
            segments[-1] = segments[-1].ljust(max_bits, '0')
        return segments

    def decompress(self, encoded):
        if not self.codebook:
            raise ValueError("Codebook not initialized.")
        decoded, buffer = '', ''
        for bit in encoded[:self.original_length]:
            buffer += bit
            if buffer in self.reverse_codebook:
                decoded += self.reverse_codebook[buffer]
                buffer = ''
        return decoded

    def decompress_segments(self, segments):
        full_encoded = ''.join(segments)
        return self.decompress(full_encoded)


class ReedSolomonCoding:
    def __init__(self, n=255, k=223):
        self.n = n
        self.k = k
        self.nsym = n - k
        self.rs = RSCodec(self.nsym)

    def encode(self, data):
        return self.rs.encode(data)

    def decode(self, data):
        try:
            decoded_tuple = self.rs.decode(data)
            return bytes(decoded_tuple[0])
        except Exception:
            return None


class Modulator:
    def __init__(self, modulation='16qam'):
        self.modulation = modulation
        self.bits_per_symbol = {'16qam': 4, '64qam': 6}[modulation]
        self.constellation = self._create_constellation()
        self.constellation_points = np.array(list(self.constellation.values()))
        self.constellation_keys = list(self.constellation.keys())

    def _create_constellation(self):
        if self.modulation == '16qam':
            constellation = {f'{i:04b}': (x + y * 1j) for i, (x, y) in
                             enumerate([(x, y) for x in [-3, -1, 1, 3] for y in
                                        [-3, -1, 1, 3]])}
        elif self.modulation == '64qam':
            constellation = {f'{i:06b}': (x + y * 1j) for i, (x, y) in
                             enumerate(
                                 [(x, y) for x in [-7, -5, -3, -1, 1, 3, 5, 7]
                                  for y in [-7, -5, -3, -1, 1, 3, 5, 7]])}
        else:
            raise ValueError(f"Unsupported modulation: {self.modulation}")
        points = np.array(list(constellation.values()))
        avg_power = np.mean(np.abs(points) ** 2)
        points = points / np.sqrt(avg_power)
        return {k: points[i] for i, k in enumerate(constellation.keys())}

    def modulate(self, bits):
        if len(bits) % self.bits_per_symbol:
            bits += '0' * (
                    self.bits_per_symbol - len(bits) % self.bits_per_symbol)
        return np.array(
            [self.constellation[bits[i:i + self.bits_per_symbol]] for i in
             range(0, len(bits), self.bits_per_symbol)])

    def demodulate(self, received):
        distances = np.abs(
            self.constellation_points[None, :] - received[:, None])
        nearest_idx = np.argmin(distances, axis=1)
        return ''.join(self.constellation_keys[idx] for idx in nearest_idx)


def strip_special_tokens(text):
    special_tokens = ["<START>", "<END>", "<PAD>", "<UNK>"]
    for token in special_tokens:
        text = text.replace(token, "")
    return ' '.join(text.split())


def data_inspection(dataset, num_samples=100, target_snr=18):
    """Inspect dataset and suggest RS n and k based on compressed length and estimated BER."""
    huffman_coder = HuffmanCoding()
    modulator = Modulator(modulation='64qam')
    channel = Channels()
    lengths = []
    bers = []

    print(f"Analyzing {min(num_samples, len(dataset))} samples...")
    with tqdm(total=min(num_samples, len(dataset)),
              desc="Data Inspection Progress") as pbar:
        for _ in range(min(num_samples, len(dataset))):
            idx = random.randint(0, len(dataset) - 1)
            sample = dataset[idx]
            if isinstance(sample, torch.Tensor):
                sample = sample.tolist()
            text = strip_special_tokens(StoT.sequence_to_text(sample))
            compressed = huffman_coder.compress([text])
            lengths.append(len(compressed))

            symbols = modulator.modulate(compressed)
            symbols_ri = np.stack((symbols.real, symbols.imag), axis=-1)
            rx_sig_tensor, _ = channel.Rayleigh(
                torch.tensor(symbols_ri, dtype=torch.float32).to(device),
                SNR_to_noise(target_snr))
            received = rx_sig_tensor.cpu().numpy()[:,
                       0] + 1j * rx_sig_tensor.cpu().numpy()[:, 1]
            received_bits = modulator.demodulate(received)[:len(compressed)]
            errors = sum(a != b for a, b in zip(compressed, received_bits))
            ber = errors / len(compressed)
            bers.append(ber)
            pbar.update(1)

    avg_length = np.mean(lengths)
    max_length = max(lengths)
    avg_ber = np.mean(bers)
    max_ber = max(bers)

    print("\nData Inspection Results:")
    print(f"Number of Samples Analyzed: {len(lengths)}")
    print(f"Average Compressed Length: {avg_length:.2f} bits")
    print(f"Max Compressed Length: {max_length} bits")
    print(f"Average BER at SNR {target_snr} dB: {avg_ber:.6f}")
    print(f"Max BER at SNR {target_snr} dB: {max_ber:.6f}")

    suggested_k = int(np.ceil(max_length / 8))
    max_bits = suggested_k * 8
    expected_errors = int(max_ber * max_bits * 1.5)
    required_symbol_corrections = int(np.ceil(expected_errors / 8))
    suggested_n = min(255, suggested_k + 2 * required_symbol_corrections)
    if suggested_n <= suggested_k:
        suggested_n = suggested_k + 64
        print("Warning: Suggested n too small, increased for more correction.")

    print(f"Suggested RS Parameters: n={suggested_n}, k={suggested_k}")
    print(
        f"Error Correction Capability: {(suggested_n - suggested_k) // 2} symbols "
        f"({(suggested_n - suggested_k) * 4} bits)")
    return suggested_n, suggested_k


def debug_single_sample(dataloader, snrs, modulation='64qam', rs_n=255,
                        rs_k=223):
    """Debug function modified to work with DataLoader and exclude BLEU scores."""
    modulator = Modulator(modulation=modulation)
    channel = Channels()
    huffman_coder = HuffmanCoding()
    rs_coder = ReedSolomonCoding(n=rs_n, k=rs_k)

    # Since batch_size=1, get a single batch (one sample) from the DataLoader
    batch_iter = iter(dataloader)
    batch = next(batch_iter)  # Get the first batch
    sample = batch[
        0]  # DataLoader with batch_size=1 returns a batch of 1 sample
    if isinstance(sample, torch.Tensor):
        sample = sample.tolist()

    # Use the first SNR value from snrs list
    snr = snrs[0] if isinstance(snrs, list) else snrs

    input_text = StoT.sequence_to_text(sample)
    cleaned_input = strip_special_tokens(input_text)
    print(f"\nDebugging Random Sample:")
    print(f"Input Text: {cleaned_input}")

    max_bits = rs_k * 8
    segments = huffman_coder.segment_compress(cleaned_input, max_bits)
    print(f"Number of Segments: {len(segments)}")
    print(f"Compressed Segments (first 50 bits each):")
    for i, seg in enumerate(segments):
        print(f"  Segment {i}: {seg[:50]}... (length: {len(seg)} bits)")

    received_bits_all = ''
    for i, segment in enumerate(segments):
        print(f"\nProcessing Segment {i}:")
        if len(segment) > max_bits:
            segment = segment[:max_bits]
        compressed_bytes = int(segment, 2).to_bytes(rs_k, byteorder='big')
        encoded = rs_coder.encode(compressed_bytes)
        encoded_bits = ''.join(f'{byte:08b}' for byte in encoded)
        print(
            f"  RS Encoded Bits (first 50): {encoded_bits[:50]}... (length: {len(encoded_bits)})")

        symbols = modulator.modulate(encoded_bits)
        print(f"  Modulated Symbols (first 5): {symbols[:5]}")

        symbols_ri = np.stack((symbols.real, symbols.imag), axis=-1)
        rx_sig_tensor, _ = channel.Rayleigh(
            torch.tensor(symbols_ri, dtype=torch.float32).to(device),
            SNR_to_noise(snr))
        received = rx_sig_tensor.cpu().numpy()[:,
                   0] + 1j * rx_sig_tensor.cpu().numpy()[:, 1]
        print(f"  Received Symbols (first 5): {received[:5]}")

        received_bits = modulator.demodulate(received)[:len(encoded_bits)]
        errors = sum(a != b for a, b in zip(encoded_bits, received_bits))
        ber = errors / len(encoded_bits)
        print(
            f"  Demodulated Bits (first 50): {received_bits[:50]}... (BER: {ber:.6f})")

        received_bytes = int(received_bits, 2).to_bytes(rs_n, byteorder='big')
        decoded = rs_coder.decode(received_bytes)
        if decoded:
            decoded_bits = ''.join(f'{byte:08b}' for byte in decoded)[
                           :len(segment)]
            print(f"  RS Decoded Bits (first 50): {decoded_bits[:50]}...")
            received_bits_all += decoded_bits
        else:
            print("  RS Decoding Failed")
            received_bits_all = ''
            break

    output_text = huffman_coder.decompress_segments(
        [received_bits_all]) if received_bits_all else ""
    print(f"\nOutput Text: {output_text}")
    return {}


def batch_test_huffman_rs(dataloader, snrs, modulation='64qam', rs_n=255,
                          rs_k=63, args=None):
    """Batch test Huffman + RS coding with sentence similarity calculation, BLEU disabled."""
    modulator = Modulator(modulation=modulation)
    channel = Channels()
    similarity = Similarity(batch_size=4)
    results = {snr: {'similarity_scores': [], 'total': 0} for snr in snrs}

    # Setup for evaluation scores
    eval_dir = "evaluation_result/huffman + rs-Rayleigh"
    os.makedirs(eval_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    scores_file = os.path.join(eval_dir,
                               f"scores_huffman + rs_similarity_{timestamp}.csv")

    # Print contents of the evaluation directory
    try:
        eval_dir_contents = os.listdir(eval_dir)
        print(
            f"Contents of evaluation directory '{eval_dir}' before saving: {eval_dir_contents}")
    except Exception as e:
        print(f"Error accessing evaluation directory '{eval_dir}': {e}")

    # Start total timing
    total_start_time = time.time()

    # Collect all scores to save at once
    all_scores = []

    for snr in snrs:
        print(f"\nBatch Testing at SNR {snr} dB...")
        snr_start_time = time.time()
        all_cleaned_inputs = []
        all_output_texts = []

        with tqdm(total=len(dataloader), desc=f"SNR {snr} dB Progress",
                  unit="batch") as pbar:
            for batch_idx, batch in enumerate(dataloader):
                input_sentences = [StoT.sequence_to_text(sent.tolist()) for sent
                                   in batch.cpu()]
                for sample_idx, input_text in enumerate(input_sentences):
                    cleaned_input = strip_special_tokens(input_text)
                    huffman_coder = HuffmanCoding()
                    compressed = huffman_coder.compress([cleaned_input])
                    compressed_length = len(compressed)
                    max_bits = 504 if compressed_length <= 504 else 1008
                    rs_k_dynamic = 63 if compressed_length <= 504 else 126
                    segments = huffman_coder.segment_compress(cleaned_input,
                                                              max_bits)
                    rs_coder = ReedSolomonCoding(n=rs_n, k=rs_k_dynamic)
                    received_bits_all = ''

                    for seg_idx, segment in enumerate(segments):
                        original_len = len(segment)
                        if len(segment) > max_bits:
                            segment = segment[:max_bits]
                        if len(segment) < max_bits:
                            segment = segment + '0' * (max_bits - len(segment))

                        segment_int = int(segment, 2)
                        byte_length = (max_bits + 7) // 8
                        compressed_bytes = segment_int.to_bytes(byte_length,
                                                                byteorder='big')
                        if len(compressed_bytes) > rs_k_dynamic:
                            compressed_bytes = compressed_bytes[-rs_k_dynamic:]
                        elif len(compressed_bytes) < rs_k_dynamic:
                            compressed_bytes = b'\x00' * (rs_k_dynamic - len(
                                compressed_bytes)) + compressed_bytes

                        encoded = rs_coder.encode(compressed_bytes)
                        encoded_bits = ''.join(
                            f'{byte:08b}' for byte in encoded)

                        symbols = modulator.modulate(encoded_bits)
                        symbols_ri = np.stack((symbols.real, symbols.imag),
                                              axis=-1)
                        rx_sig_tensor, _ = channel.Rayleigh(
                            torch.tensor(symbols_ri, dtype=torch.float32).to(
                                device),
                            SNR_to_noise(snr))
                        received = rx_sig_tensor.cpu().numpy()[:,
                                   0] + 1j * rx_sig_tensor.cpu().numpy()[:, 1]
                        received_bits = modulator.demodulate(received)[
                                        :len(encoded_bits)]

                        received_bytes = int(received_bits, 2).to_bytes(rs_n,
                                                                        byteorder='big')
                        decoded = rs_coder.decode(received_bytes)

                        if decoded:
                            decoded_bits = ''.join(
                                f'{byte:08b}' for byte in decoded)[
                                           :original_len]
                            received_bits_all += decoded_bits
                        else:
                            received_bits_all = ''
                            break

                    output_text = huffman_coder.decompress_segments(
                        [received_bits_all]) if received_bits_all else ""
                    all_cleaned_inputs.append(cleaned_input)
                    all_output_texts.append(output_text)
                    results[snr]['total'] += 1
                pbar.update(1)
                torch.cuda.empty_cache()

        # Compute similarity for this SNR
        if all_cleaned_inputs and all_output_texts:
            print("Computing similarity...")
            print("Memory status before computing similarity:")
            memory_status()
            similarities = similarity.compute_similarity(all_cleaned_inputs,
                                                         all_output_texts)
            mean_similarity = np.mean(similarities)
            print("Memory status after computing similarity:")
            memory_status()
        else:
            mean_similarity = 0.0

        results[snr][
            'similarity_scores'] = similarities if all_cleaned_inputs else []
        print(f"SNR {snr} dB: Similarity = {mean_similarity:.4f}")

        # End timing for this SNR
        snr_end_time = time.time()
        snr_duration = snr_end_time - snr_start_time
        print(f"Time taken for SNR {snr} dB: {snr_duration:.2f} seconds")

        # Collect scores for this SNR
        all_scores.append({
            'SNR': snr,
            'similarity_score': mean_similarity,
            'method': "huffman + rs",
            'total_samples': results[snr]['total']
        })

    # End total timing
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    # Save all evaluation scores at once
    if all_scores:
        try:
            print("Collected scores:", all_scores)
            scores_df = pd.DataFrame(all_scores)
            scores_df.to_csv(scores_file, index=False)
            print(f"\nResults saved to: {scores_file}")
            saved_df = pd.read_csv(scores_file)
            print(
                f"Contents of saved scores file:\n{saved_df.to_string(index=False)}")
        except Exception as e:
            print(f"Error saving evaluation scores to {scores_file}: {e}")

    # Print contents of the evaluation directory again after saving
    try:
        eval_dir_contents = os.listdir(eval_dir)
        print(
            f"Contents of evaluation directory '{eval_dir}' after saving: {eval_dir_contents}")
    except Exception as e:
        print(
            f"Error accessing evaluation directory '{eval_dir}' after saving: {e}")
    print(f"Total time taken for batch test: {total_duration:.2f} seconds")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Huffman + RS Test Script with Similarity")
    parser.add_argument('--channel', default='Rayleigh', type=str,
                        help='Channel type')
    parser.add_argument('--MAX-LENGTH', default=30, type=int,
                        help='Maximum sentence length')
    parser.add_argument('--MIN-LENGTH', default=4, type=int,
                        help='Minimum sentence length')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size')
    parser.add_argument('--d_model', default=512, type=int,
                        help='Model dimension (placeholder)')
    parser.add_argument('--num_layers', default=6, type=int,
                        help='Number of layers (placeholder)')
    parser.add_argument('--num_heads', default=8, type=int,
                        help='Number of heads (placeholder)')
    parser.add_argument('--dff', default=2048, type=int,
                        help='Feed-forward dimension (placeholder)')
    parser.add_argument('--epochs', default=20, type=int,
                        help='Epochs (placeholder)')
    args = parser.parse_args()

    test_eur = EurDataset('test')
    snrs = [30]
    modulation = '64qam'

    batch_loader = DataLoader(test_eur, batch_size=1,
                              num_workers=0,
                              pin_memory=(device.startswith('cuda')),
                              collate_fn=collate_data)

    print("Starting debug test with dynamic RS...")
    debug_single_sample(batch_loader, snrs, modulation, rs_n=255, rs_k=63)

    # print("\nStarting batch test with dynamic RS and similarity calculation...")
    # batch_test_huffman_rs(batch_loader, snrs, modulation, rs_n=255, rs_k=63,
    #                       args=args)
