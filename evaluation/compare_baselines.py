#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File: compare_baselines.py
@Description: Tests traditional source/channel coding on sampled sentences with variable SNR and symbols per word.
"""

import json
import random
import numpy as np
from dataset import EurDataset, collate_data
import huffman
import string
import brotli
from reedsolo import RSCodec
from utils import BleuScore, SNR_to_noise, SeqtoText
from torch.utils.data import DataLoader
import torch
from transformers import BertModel, BertTokenizer
from sklearn.preprocessing import normalize as sk_normalize

# Device setup
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Vocabulary
VOCAB_FILE = './data/vocab.json'
print("Loading vocabulary...")
vocab = json.load(open(VOCAB_FILE, 'rb'))
token_to_idx = vocab['token_to_idx']
idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
pad_idx = token_to_idx["<PAD>"]
start_idx = token_to_idx["<START>"]
end_idx = token_to_idx["<END>"]
unk_idx = token_to_idx["<UNK>"]
StoT = SeqtoText(token_to_idx, end_idx)


# Sampling Function
def sample_dataset_sentences(dataset: EurDataset, num_samples: int,
                             batch_size: int) -> DataLoader:
    """Sample sentences from dataset with proper batching and padding."""
    sampled_indices = random.sample(range(len(dataset)),
                                    min(num_samples, len(dataset)))
    sampled_subset = torch.utils.data.Subset(dataset, sampled_indices)
    return DataLoader(
        sampled_subset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_data
    )


# Source Coding
def huffman_compress(sentences):
    text = ' '.join(sentences)
    freq = {char: text.count(char) for char in set(text)}
    codebook = huffman.codebook(freq.items())
    encoded = ''.join(codebook[char] for char in text)
    return encoded, codebook


def huffman_decompress(encoded, codebook):
    reverse_codebook = {v: k for k, v in codebook.items()}
    decoded, buffer = '', ''
    for bit in encoded:
        buffer += bit
        if buffer in reverse_codebook:
            decoded += reverse_codebook[buffer]
            buffer = ''
    return decoded


def fixed_5bit_compress(sentences):
    alphabet = string.ascii_lowercase + ' ,.?!'
    char_to_bin = {char: f'{i:05b}' for i, char in enumerate(alphabet)}
    text = ' '.join(sentences).lower()
    encoded = ''.join(char_to_bin.get(char, '00000') for char in text)
    return encoded, char_to_bin


def fixed_5bit_decompress(encoded, char_to_bin):
    bin_to_char = {v: k for k, v in char_to_bin.items()}
    decoded = ''
    for i in range(0, len(encoded), 5):
        bits = encoded[i:i + 5]
        decoded += bin_to_char.get(bits, '')
    return decoded


def brotli_compress(sentences):
    text = ' '.join(sentences).encode('utf-8')
    return brotli.compress(text)


def brotli_decompress(compressed):
    try:
        return brotli.decompress(compressed).decode('utf-8')
    except:
        return ''


# Replace rs_encode with this:
def rs_encode(data):
    rs = RSCodec(12)  # RS(42,30), 42-30 = 12 symbols of error correction
    return rs.encode(data)


# Replace rs_decode with this:
def rs_decode(data):
    rs = RSCodec(12)
    try:
        decoded_tuple = rs.decode(data)
        return bytes(decoded_tuple[0])
    except Exception as e:
        print(f"RS Decode Error: {e}")
        return None


def turbo_encode(data):
    from commpy.channelcoding import Trellis, conv_encode
    trellis = Trellis(np.array([2]), np.array([[7, 5]]))
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    parity1 = conv_encode(bits, trellis)[len(bits):]
    block_size = 10
    bits_padded = np.pad(bits, (
        0,
        block_size - len(bits) % block_size if len(bits) % block_size else 0),
                         mode='constant')
    interleaved = bits_padded.reshape(-1, block_size).T.flatten()[:len(bits)]
    parity2 = conv_encode(interleaved, trellis)[len(bits):]
    encoded_bits = np.concatenate([bits, parity1, parity2])
    return np.packbits(encoded_bits).tobytes()


def turbo_decode(data, iterations=5):
    from commpy.channelcoding import Trellis, viterbi_decode
    trellis = Trellis(np.array([2]), np.array([[7, 5]]))
    received_bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    expected_length = len(received_bits) // 3
    if len(received_bits) < 3 * expected_length:
        return None
    systematic = received_bits[:expected_length]
    parity1 = received_bits[expected_length:2 * expected_length]
    parity2 = received_bits[2 * expected_length:3 * expected_length]
    block_size = 10
    systematic_padded = np.pad(systematic, (0, block_size - len(
        systematic) % block_size if len(systematic) % block_size else 0),
                               mode='constant')
    interleaved_systematic = systematic_padded.reshape(-1,
                                                       block_size).T.flatten()[
                             :len(systematic)]
    extrinsic_info = np.zeros_like(systematic, dtype=float)
    for _ in range(iterations):
        received1 = np.concatenate([systematic + extrinsic_info, parity1])
        decoded1 = viterbi_decode(received1.astype(float), trellis)
        extrinsic_info_interleaved = decoded1 - systematic
        interleaved_extrinsic = np.zeros_like(systematic)
        interleaved_extrinsic[
        :len(extrinsic_info_interleaved)] = extrinsic_info_interleaved
        received2 = np.concatenate(
            [interleaved_systematic + interleaved_extrinsic, parity2])
        decoded2 = viterbi_decode(received2.astype(float), trellis)
        extrinsic_info = decoded2[:len(systematic)] - interleaved_systematic
    decoded_bits = decoded2[:len(systematic)]
    return np.packbits(decoded_bits).tobytes()


# Modulation/Demodulation
# Replace modulate with this:
def modulate(bits, modulation='8qam'):
    bits_per_symbol = {'8qam': 3, '64qam': 6, '128qam': 7}[modulation]
    if modulation == '8qam':
        constellation = {'000': (-3 - 3j), '001': (-3 - 1j), '010': (-1 - 3j),
                         '011': (-1 - 1j),
                         '100': (3 + 3j), '101': (3 + 1j), '110': (1 + 3j),
                         '111': (1 + 1j)}
    elif modulation == '64qam':
        constellation = {f'{i:06b}': (x + y * 1j) for i, (x, y) in enumerate(
            [(x, y) for x in [-7, -5, -3, -1, 1, 3, 5, 7] for y in
             [-7, -5, -3, -1, 1, 3, 5, 7]])}
    elif modulation == '128qam':
        constellation = {f'{i:07b}': (x + y * 1j) for i, (x, y) in enumerate(
            [(x, y) for x in [-7, -5, -3, -1, 1, 3, 5, 7] for y in
             [-7, -5, -3, -1, 1, 3, 5, 7]])[:128]}

    # Normalize to unit average power
    points = np.array(list(constellation.values()))
    avg_power = np.mean(np.abs(points) ** 2)
    points = points / np.sqrt(avg_power)
    constellation = {k: points[i] for i, k in enumerate(constellation.keys())}

    if len(bits) % bits_per_symbol:
        bits += '0' * (bits_per_symbol - len(bits) % bits_per_symbol)
    return np.array([constellation[bits[i:i + bits_per_symbol]] for i in
                     range(0, len(bits), bits_per_symbol)])


# Replace demodulate with this:
def demodulate(received, modulation='8qam'):
    if modulation == '8qam':
        constellation = {(-3 - 3j): '000', (-3 - 1j): '001', (-1 - 3j): '010',
                         (-1 - 1j): '011',
                         (3 + 3j): '100', (3 + 1j): '101', (1 + 3j): '110',
                         (1 + 1j): '111'}
    elif modulation == '64qam':
        constellation = {(x + y * 1j): f'{i:06b}' for i, (x, y) in enumerate(
            [(x, y) for x in [-7, -5, -3, -1, 1, 3, 5, 7] for y in
             [-7, -5, -3, -1, 1, 3, 5, 7]])}
    elif modulation == '128qam':
        constellation = {(x + y * 1j): f'{i:07b}' for i, (x, y) in enumerate(
            [(x, y) for x in [-7, -5, -3, -1, 1, 3, 5, 7] for y in
             [-7, -5, -3, -1, 1, 3, 5, 7]])[:128]}

    # Normalize to unit average power
    points = np.array(list(constellation.keys()))
    avg_power = np.mean(np.abs(points) ** 2)
    points = points / np.sqrt(avg_power)
    constellation = {points[i]: v for i, v in enumerate(constellation.values())}

    return ''.join(constellation[min(constellation.keys(),
                                     key=lambda p: np.abs(received[i] - p))] for
                   i in range(len(received)))


# Similarity Class
class Similarity:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.model = BertModel.from_pretrained('bert-base-cased')
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def prepare_sentences(self, sentences, max_length=32):
        encoded = self.tokenizer(sentences, padding=True, truncation=True,
                                 max_length=max_length, return_tensors='pt')
        return encoded['input_ids'].to(self.device), encoded[
            'attention_mask'].to(self.device)

    def get_sentence_embeddings(self, sentences):
        input_ids, _ = self.prepare_sentences(sentences)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-2]
            return torch.sum(hidden_states, dim=1).cpu().numpy()

    def compute_similarity(self, real, predicted):
        if not real or not predicted:
            return []
        real_embeddings = self.get_sentence_embeddings(real)
        pred_embeddings = self.get_sentence_embeddings(predicted)
        real_norm = sk_normalize(real_embeddings, norm='max', axis=0)
        pred_norm = sk_normalize(pred_embeddings, norm='max', axis=0)
        dot = np.sum(real_norm * pred_norm, axis=1)
        a = np.sqrt(np.sum(real_norm ** 2, axis=1))
        b = np.sqrt(np.sum(pred_norm ** 2, axis=1))
        return (dot / (a * b)).tolist()


# Test Function
# Replace test_sample_sentences with this:
def test_sample_sentences(source_coding, channel_coding, dataloader: DataLoader,
                          snr: float, modulation: str,
                          similarity_calculator: Similarity) -> None:
    """Test traditional coding on sampled sentences with specific configurations from the paper."""
    # Define method-specific parameters based on the paper
    method_params = {
        ('huffman', 'rs', '64qam'): {'rs_n': 42, 'rs_k': 30, 'max_bits': 240},
        ('5bit', 'rs', '64qam'): {'rs_n': 54, 'rs_k': 42, 'max_bits': 336},
        ('huffman', 'turbo', '64qam'): {'max_bits': 240},
        ('5bit', 'turbo', '128qam'): {'max_bits': 336},
        ('brotli', 'turbo', '8qam'): {'max_bits': 120},
    }

    # Get method-specific parameters
    key = (source_coding, channel_coding, modulation)
    if key not in method_params:
        print(f"Configuration {key} not supported.")
        return
    params = method_params[key]
    max_bits = params['max_bits']
    bits_per_symbol = {'8qam': 3, '64qam': 6, '128qam': 7}[modulation]
    total_symbols = (params[
                         'rs_n'] * 8) // bits_per_symbol if channel_coding == 'rs' else max_bits // bits_per_symbol

    print(
        f"\nTesting {len(dataloader.dataset)} sentences with {source_coding} + {channel_coding} ({modulation}) at SNR {snr} dB, {total_symbols} symbols/sentence ({max_bits} bits max)")
    bleu_calc = BleuScore(1, 0, 0, 0)

    for batch_idx, input_batch in enumerate(dataloader):
        input_batch = input_batch.to(device) if isinstance(input_batch,
                                                           torch.Tensor) else \
            input_batch[0].to(device)
        input_sentences = [StoT.sequence_to_text(sent.cpu().numpy().tolist())
                           for sent in input_batch]

        # Source Coding
        if source_coding == 'huffman':
            compressed, codebook = huffman_compress(input_sentences)
            print(f"Original compressed bits length: {len(compressed)}")
            if len(compressed) > max_bits:
                compressed_bits = compressed[:max_bits]
            else:
                compressed_bits = compressed + '0' * (
                        max_bits - len(compressed))
            print(f"Adjusted compressed bits length: {len(compressed_bits)}")
            compressed_bytes = int(compressed_bits, 2).to_bytes(
                (len(compressed_bits) + 7) // 8, 'big')
            print(f"Compressed bytes length: {len(compressed_bytes)}")
        else:
            print(
                f"Source coding {source_coding} not implemented in this test.")
            return

        # Channel Coding
        if channel_coding == 'rs':
            rs = RSCodec(params['rs_n'] - params[
                'rs_k'])  # e.g., 42-30 = 12 symbols of error correction
            encoded = rs.encode(compressed_bytes)
        else:
            encoded = turbo_encode(compressed_bytes)
        bits = ''.join(f'{byte:08b}' for byte in encoded)
        print(f"Encoded bits length: {len(bits)}")

        # Modulation
        symbols = modulate(bits, modulation)
        print(f"Number of symbols: {len(symbols)}")

        # Convert symbols to real/imag pairs for channel application
        symbols_ri = np.stack((symbols.real, symbols.imag),
                              axis=-1)  # Shape: (56, 2)

        # Rayleigh fading: single coefficient per transmission
        h_real = np.random.normal(0, np.sqrt(0.5), size=1)  # E[|h|^2] = 1
        h_imag = np.random.normal(0, np.sqrt(0.5), size=1)
        h = np.array(
            [[h_real, -h_imag], [h_imag, h_real]]).squeeze()  # Shape: (2, 2)

        # Apply fading
        tx_sig = np.dot(symbols_ri, h)  # Shape: (56, 2)

        # Add AWGN
        noise_std = SNR_to_noise(snr)
        noise = np.random.normal(0, noise_std, tx_sig.shape)  # Shape: (56, 2)
        rx_sig = tx_sig + noise

        # Channel estimation (perfect CSI)
        h_inv = np.linalg.inv(h)  # Shape: (2, 2)
        rx_sig = np.dot(rx_sig, h_inv)  # Shape: (56, 2)

        # Convert back to complex symbols
        received = rx_sig[:, 0] + 1j * rx_sig[:, 1]  # Shape: (56,)

        # Demodulate
        received_bits = demodulate(received, modulation)
        if len(received_bits) > len(bits):
            received_bits = received_bits[:len(bits)]
        elif len(received_bits) < len(bits):
            received_bits += '0' * (len(bits) - len(received_bits))
        print(
            f"Received bits length: {len(received_bits)} (adjusted to match encoded)")

        # Calculate Bit Error Rate (BER)
        errors = sum(a != b for a, b in zip(bits, received_bits))
        ber = errors / len(bits)
        print(
            f"Bit Error Rate (BER): {ber:.6f} ({errors} errors out of {len(bits)} bits)")

        # Log error positions to debug symbol errors
        error_positions = [i for i, (a, b) in
                           enumerate(zip(bits, received_bits)) if a != b]
        # Map bit errors to symbols (8 bits per symbol)
        affected_symbols = set(i // 8 for i in error_positions)
        print(
            f"Number of affected symbols: {len(affected_symbols)} (out of {len(bits) // 8} symbols)")

        # Decoding
        try:
            received_bytes = int(received_bits, 2).to_bytes(
                (len(received_bits) + 7) // 8, 'big')
            print(f"Received bytes length: {len(received_bytes)}")
        except ValueError as e:
            print(f"Error: Failed to convert received bits to bytes - {e}")
            received_bytes = None

        if channel_coding == 'rs':
            rs = RSCodec(params['rs_n'] - params['rs_k'])
            decoded = rs.decode(received_bytes)[0] if received_bytes else None
        else:
            decoded = turbo_decode(received_bytes) if received_bytes else None

        if decoded is None:
            output_sentences = [""] * len(input_sentences)
            print("Error: Decoding failed")
        else:
            decoded_bits = ''.join(f'{byte:08b}' for byte in decoded)
            print(f"Decoded bits length: {len(decoded_bits)}")
            reconstructed = huffman_decompress(decoded_bits,
                                               codebook) if source_coding == 'huffman' else ""
            output_sentences = [reconstructed] if reconstructed else [""] * len(
                input_sentences)

        # Metrics and Printing
        for i, (input_text, output_text) in enumerate(
                zip(input_sentences, output_sentences)):
            bleu = bleu_calc.compute_blue_score([input_text], [output_text])[0]
            sim = similarity_calculator.compute_similarity([input_text],
                                                           [output_text])[0]
            print(
                f"\nTest {batch_idx * len(input_sentences) + i + 1}/{len(dataloader.dataset)}:")
            print(f"Input: {input_text}")
            print(f"Output: {output_text}")
            print(f"BLEU Score: {bleu:.4f}")
            print(f"Similarity Score: {sim:.4f}")


# Main Execution
if __name__ == '__main__':
    print("Loading test dataset...")
    test_eur = EurDataset('test')
    print(f"Dataset size: {len(test_eur)} sentences")

    # Test parameters
    combinations = [('huffman', 'rs'),
                    ('huffman', 'turbo')]  # Test both RS and Turbo
    snrs = [18]  # Test only at SNR 18 dB
    modulations = ['64qam']  # Test only with 64-QAM as specified
    similarity = Similarity()

    # Sample and test
    sample_loader = sample_dataset_sentences(test_eur, num_samples=1,
                                             batch_size=1)
    for source_coding, channel_coding in combinations:
        for modulation in modulations:
            for snr in snrs:
                test_sample_sentences(source_coding, channel_coding,
                                      sample_loader, snr, modulation,
                                      similarity)
