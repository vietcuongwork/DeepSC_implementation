#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File: fixed5bit_rs_test.py
@Description: Tests 5-bit fixed-length coding + RS coding with 64-QAM, with dataset inspection and single-sentence debug using sentence similarity.
"""
import time
from datetime import datetime
import json
import os
import numpy as np
import pandas as pd
from dataset import EurDataset, collate_data
from reedsolo import RSCodec
from huffman_rs_test import strip_special_tokens
from utils import SNR_to_noise, SeqtoText, Channels
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import tqdm as tqdm_module
import argparse
import math
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
pad_idx = token_to_idx["<PAD>"]
start_idx = token_to_idx["<START>"]
end_idx = token_to_idx["<END>"]
unk_idx = token_to_idx["<UNK>"]
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


class FixedLengthCoding:
    def __init__(self, bit_length=5):
        self.bit_length = bit_length
        self.char_to_code = {}
        self.code_to_char = {}
        self.original_length = 0

    def compress(self, sentences):
        text = ' '.join(sentences)
        unique_chars = set(text)
        if len(unique_chars) > 2 ** self.bit_length:
            raise ValueError(
                f"Too many unique characters ({len(unique_chars)}) for {self.bit_length}-bit coding.")
        self.char_to_code = {char: f'{i:0{self.bit_length}b}' for i, char in
                             enumerate(unique_chars)}
        self.code_to_char = {code: char for char, code in
                             self.char_to_code.items()}
        encoded = ''.join(self.char_to_code[char] for char in text)
        self.original_length = len(encoded)
        return encoded

    def segment_compress(self, sentences, max_bits):
        compressed = self.compress(sentences)
        segments = [compressed[i:i + max_bits] for i in
                    range(0, len(compressed), max_bits)]
        if len(segments[-1]) < max_bits:
            segments[-1] = segments[-1].ljust(max_bits, '0')
        return segments

    def decompress(self, encoded):
        if not self.code_to_char:
            raise ValueError("Codebook not initialized.")
        decoded = ''
        for i in range(0, self.original_length, self.bit_length):
            code = encoded[i:i + self.bit_length]
            decoded += self.code_to_char.get(code, '?')
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
    def __init__(self, modulation='64qam'):
        self.modulation = modulation
        self.bits_per_symbol = {'64qam': 6}[modulation]
        self.constellation = self._create_constellation()
        self.constellation_points = np.array(list(self.constellation.values()))
        self.constellation_keys = list(self.constellation.keys())
        self.signal_power = np.mean(np.abs(self.constellation_points) ** 2)

    def _create_constellation(self):
        constellation = {f'{i:06b}': (x + y * 1j) for i, (x, y) in
                         enumerate([(x, y) for x in [-7, -5, -3, -1, 1, 3, 5, 7]
                                    for y in [-7, -5, -3, -1, 1, 3, 5, 7]])}
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


def inspect_dataset(dataset, vocab_file='./data/vocab.json', sample_size=None):
    with open(vocab_file, 'rb') as f:
        vocab = json.load(f)
    token_to_idx = vocab['token_to_idx']
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    end_idx = token_to_idx["<END>"]
    StoT = SeqtoText(token_to_idx, end_idx)

    fixed_coder = FixedLengthCoding(bit_length=5)
    num_samples = sample_size if sample_size else len(dataset)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=0,
                            collate_fn=collate_data)
    print(
        f"Inspecting dataset with {len(dataset)} total samples, analyzing {num_samples}...")

    sentence_lengths = []
    compressed_lengths = []

    for i, batch in enumerate(
            tqdm(dataloader, desc="Processing batches", unit="batch")):
        if i * 32 >= num_samples:
            break
        input_sentences = [StoT.sequence_to_text(sent.tolist()) for sent in
                           batch.cpu()]
        for sent in input_sentences:
            cleaned_sent = strip_special_tokens(sent)
            tokens = cleaned_sent.split()
            sentence_lengths.append(len(tokens))
            if tokens:
                try:
                    compressed_bits = fixed_coder.compress([cleaned_sent])
                    compressed_lengths.append(len(compressed_bits))
                except ValueError as e:
                    print(f"Skipping sentence due to {e}")
                    continue

    total_sentences = len(sentence_lengths)
    if total_sentences == 0:
        print("No valid sentences found in dataset.")
        return None

    total_tokens = sum(sentence_lengths)
    avg_sentence_length = np.mean(sentence_lengths)
    sentences_in_range = sum(
        1 for length in sentence_lengths if 4 <= length <= 30)
    avg_compressed_length = np.mean(
        compressed_lengths) if compressed_lengths else 0
    max_compressed_length = max(compressed_lengths) if compressed_lengths else 0
    min_compressed_length = min(compressed_lengths) if compressed_lengths else 0
    percentile_95_compressed = np.percentile(compressed_lengths,
                                             95) if compressed_lengths else 0

    print(
        f"\nDataset Inspection Results (based on {total_sentences} sentences):")
    print(f"Total tokens: {total_tokens}")
    print(f"Average sentence length (tokens): {avg_sentence_length:.1f}")
    print(
        f"Sentences with 4-30 tokens: {sentences_in_range} ({sentences_in_range / total_sentences * 100:.1f}%)")
    print(
        f"Average 5-bit compressed length (bits): {avg_compressed_length:.1f}")
    print(f"Max 5-bit compressed length (bits): {max_compressed_length}")
    print(f"Min 5-bit compressed length (bits): {min_compressed_length}")
    print(
        f"95th percentile compressed length (bits): {percentile_95_compressed:.1f}")

    suggested_k = int(np.ceil(percentile_95_compressed / 8))
    suggested_n = min(255, suggested_k + 60)
    if suggested_n <= suggested_k:
        suggested_n = min(255, suggested_k + 30)
    print(f"\nSuggested RS parameters:")
    print(
        f" - k = {suggested_k} (covers 95% of compressed lengths, {suggested_k * 8} bits)")
    print(
        f" - n = {suggested_n} (error correction up to {(suggested_n - suggested_k) // 2} symbols)")

    return {
        "total_sentences": total_sentences,
        "total_tokens": total_tokens,
        "avg_sentence_length": avg_sentence_length,
        "sentences_in_range": sentences_in_range,
        "avg_compressed_length": avg_compressed_length,
        "max_compressed_length": max_compressed_length,
        "min_compressed_length": min_compressed_length,
        "percentile_95_compressed": percentile_95_compressed,
        "suggested_n": suggested_n,
        "suggested_k": suggested_k
    }


def sample_dataset_sentences(dataset: EurDataset, num_samples: int,
                             batch_size: int) -> DataLoader:
    sampled_indices = list(range(len(dataset)))[:num_samples]
    sampled_subset = torch.utils.data.Subset(dataset, sampled_indices)
    return DataLoader(sampled_subset, batch_size=batch_size, num_workers=0,
                      pin_memory=(device.startswith('cuda')),
                      collate_fn=collate_data)


def debug_5bit_rs(dataloader: DataLoader, snr: float, modulation: str,
                  rs_n=255, rs_k=63) -> None:
    fixed_coder = FixedLengthCoding(bit_length=5)
    modulator = Modulator(modulation=modulation)
    channel = Channels()
    rs_coder = ReedSolomonCoding(n=rs_n, k=rs_k)

    input_batch = next(iter(dataloader))
    input_batch = input_batch.to(device) if isinstance(input_batch,
                                                       torch.Tensor) else \
        input_batch[0].to(device)
    input_sentence = StoT.sequence_to_text(input_batch[0].tolist())
    print(
        f"\n=== Debugging Single Sentence with 5-bit + RS({rs_n},{rs_k}) ({modulation}) at SNR {snr} dB ===")
    print(f"Input Sentence: {input_sentence}")

    cleaned_input = strip_special_tokens(input_sentence)
    print(f"Cleaned Input: {cleaned_input}")

    max_bits = rs_k * 8
    try:
        segments = fixed_coder.segment_compress([cleaned_input], max_bits)
    except ValueError as e:
        print(f"Error: Compression failed - {e}")
        return
    print(f"Number of Segments: {len(segments)}")
    print(f"Compressed Segments (first 50 bits each):")
    for i, seg in enumerate(segments):
        print(f"  Segment {i}: {seg[:50]}... (length: {len(seg)} bits)")

    received_bits_all = ''
    for seg_idx, segment in enumerate(segments):
        print(f"\nProcessing Segment {seg_idx}:")

        compressed_bytes = int(segment, 2).to_bytes(rs_k, byteorder='big')
        encoded = rs_coder.encode(compressed_bytes)
        encoded_bits = ''.join(f'{byte:08b}' for byte in encoded)
        print(
            f"  Encoded Bits: {encoded_bits[:50]}... (length: {len(encoded_bits)} bits)")
        print(
            f"  Encoded Length: {len(encoded_bits)} bits (RS codeword: {rs_n * 8} bits expected)")

        symbols = modulator.modulate(encoded_bits)
        total_symbols = (rs_n * 8) // modulator.bits_per_symbol
        print(f"  Modulated Symbols: {symbols[:5]}... (first 5 symbols)")
        print(
            f"  Number of Symbols: {len(symbols)} (expected: {total_symbols})")
        symbols_ri = np.stack((symbols.real, symbols.imag), axis=-1)
        symbols_ri_tensor = torch.tensor(symbols_ri, dtype=torch.float32).to(
            device)

        noise_std = SNR_to_noise(snr)
        print(
            f"  Noise Std Dev: {noise_std:.6f}, Signal Power: {modulator.signal_power:.6f}")
        H_real = torch.normal(0, math.sqrt(1 / 2), size=[1]).to(device)
        H_imag = torch.normal(0, math.sqrt(1 / 2), size=[1]).to(device)
        H = torch.complex(H_real, H_imag)
        fading_magnitude = torch.abs(H).item()
        print(f"  Fading Magnitude: {fading_magnitude:.4f}")
        rx_sig_tensor, _ = channel.Rayleigh(symbols_ri_tensor, noise_std)
        rx_sig = rx_sig_tensor.cpu().numpy()
        received = rx_sig[:, 0] + 1j * rx_sig[:, 1]
        print(f"  Received Symbols: {received[:5]}... (first 5 symbols)")

        received_bits = modulator.demodulate(received)
        if len(received_bits) > len(encoded_bits):
            received_bits = received_bits[:len(encoded_bits)]
        elif len(received_bits) < len(encoded_bits):
            received_bits = received_bits.ljust(len(encoded_bits), '0')
        print(
            f"  Received Bits: {received_bits[:50]}... (length: {len(received_bits)} bits)")

        errors = sum(a != b for a, b in zip(encoded_bits, received_bits))
        ber = errors / len(encoded_bits)
        error_positions = [i for i, (a, b) in
                           enumerate(zip(encoded_bits, received_bits)) if
                           a != b]
        affected_symbols = len(set(i // 8 for i in error_positions))
        print(f"  Bit Errors: {errors} out of {len(encoded_bits)} bits")
        print(f"  Bit Error Rate (BER): {ber:.6f}")
        print(
            f"  Affected Symbols: {affected_symbols} out of {len(encoded_bits) // 8} (RS correction limit: {(rs_n - rs_k) // 2})")
        if error_positions:
            print(f"  First 10 Error Positions: {error_positions[:10]}")

        received_bytes = int(received_bits, 2).to_bytes(rs_n, byteorder='big')
        decoded = rs_coder.decode(received_bytes)
        if decoded:
            decoded_bits = ''.join(f'{byte:08b}' for byte in decoded)[
                           :len(segment)]
            print(
                f"  Decoded Bits: {decoded_bits[:50]}... (length: {len(decoded_bits)} bits)")
            received_bits_all += decoded_bits
        else:
            print("  Error: RS Decoding Failed for Segment")
            received_bits_all = ''
            break

    output_text = fixed_coder.decompress_segments(
        [received_bits_all]) if received_bits_all else ""
    print(f"Output Sentence: {output_text}")

    print(f"\n=== Debug Summary ===")
    print(f"Input: {cleaned_input}")
    print(f"Output: {output_text}")
    print(f"BER (last segment): {ber:.6f}")
    print(f"Affected Symbols (last segment): {affected_symbols}")
    print(
        f"Successful Decode: {'Yes' if received_bits_all else 'No'}")


def batch_test_5bit_rs(dataloader: DataLoader, snrs: list, modulation: str,
                       args, rs_n=255, rs_k=63) -> None:
    fixed_coder = FixedLengthCoding(bit_length=5)
    modulator = Modulator(modulation=modulation)
    channel = Channels()
    similarity = Similarity(batch_size=4)

    results = {
        snr: {
            'similarity_scores': [],
            'ber': [],
            'affected_symbols': [],
            'total': 0
        } for snr in snrs
    }

    eval_dir = "evaluation_result/5bit + rs-Rayleigh"
    os.makedirs(eval_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    scores_file = os.path.join(eval_dir,
                               f"scores_5bit + rs_similarity_{timestamp}.csv")

    try:
        eval_dir_contents = os.listdir(eval_dir)
        print(
            f"Contents of evaluation directory '{eval_dir}' before saving: {eval_dir_contents}")
    except Exception as e:
        print(f"Error accessing evaluation directory '{eval_dir}': {e}")

    total_start_time = time.time()
    all_scores = []

    original_tqdm_init = tqdm_module.tqdm.__init__

    def custom_tqdm_init(*args, **kwargs):
        if 'desc' in kwargs and kwargs['desc'].startswith("Batch Test SNR"):
            return original_tqdm_init(*args, **kwargs)
        return original_tqdm_init(*args, **kwargs, disable=True)

    tqdm_module.tqdm.__init__ = custom_tqdm_init

    try:
        for snr in snrs:
            print(f"\nBatch Testing at SNR {snr} dB...")
            snr_start_time = time.time()
            all_cleaned_inputs = []
            all_output_texts = []

            with tqdm(total=len(dataloader.dataset),
                      desc=f"Batch Test SNR {snr} dB",
                      leave=True) as pbar:
                for batch_idx, input_batch in enumerate(dataloader):
                    input_batch = input_batch.to(device) if isinstance(
                        input_batch,
                        torch.Tensor) else \
                        input_batch[0].to(device)
                    input_sentences = [StoT.sequence_to_text(sent.tolist()) for
                                       sent in input_batch.cpu()]

                    for sample_idx, input_text in enumerate(input_sentences):
                        cleaned_text = strip_special_tokens(input_text)
                        try:
                            compressed = fixed_coder.compress([cleaned_text])
                            compressed_length = len(compressed)
                        except ValueError as e:
                            print(f"Skipping sentence due to {e}")
                            results[snr]['total'] += 1
                            pbar.update(1)
                            continue

                        max_bits = 504 if compressed_length <= 504 else 1008
                        rs_k_dynamic = 63 if compressed_length <= 504 else 126
                        segments = fixed_coder.segment_compress([cleaned_text],
                                                                max_bits)
                        rs_coder = ReedSolomonCoding(n=rs_n, k=rs_k_dynamic)
                        received_bits_all = ''

                        for seg_idx, segment in enumerate(segments):
                            original_len = len(segment)
                            if len(segment) < max_bits:
                                segment = segment + '0' * (
                                        max_bits - len(segment))

                            segment_int = int(segment, 2)
                            byte_length = (max_bits + 7) // 8
                            compressed_bytes = segment_int.to_bytes(byte_length,
                                                                    byteorder='big')
                            if len(compressed_bytes) > rs_k_dynamic:
                                compressed_bytes = compressed_bytes[
                                                   -rs_k_dynamic:]
                            elif len(compressed_bytes) < rs_k_dynamic:
                                compressed_bytes = b'\x00' * (
                                        rs_k_dynamic - len(
                                    compressed_bytes)) + compressed_bytes

                            encoded = rs_coder.encode(compressed_bytes)
                            encoded_bits = ''.join(
                                f'{byte:08b}' for byte in encoded)

                            symbols = modulator.modulate(encoded_bits)
                            symbols_ri = np.stack((symbols.real, symbols.imag),
                                                  axis=-1)
                            symbols_ri_tensor = torch.tensor(symbols_ri,
                                                             dtype=torch.float32).to(
                                device)
                            rx_sig_tensor, _ = channel.Rayleigh(
                                symbols_ri_tensor,
                                SNR_to_noise(snr))
                            rx_sig = rx_sig_tensor.cpu().numpy()
                            received = rx_sig[:, 0] + 1j * rx_sig[:, 1]
                            received_bits = modulator.demodulate(received)
                            received_bits = received_bits[
                                            :len(encoded_bits)].ljust(
                                len(encoded_bits), '0')

                            errors = sum(
                                a != b for a, b in
                                zip(encoded_bits, received_bits))
                            ber = errors / len(encoded_bits)
                            symbol_errors = len(set(
                                i // 8 for i in range(len(encoded_bits)) if
                                encoded_bits[i] != received_bits[i]))
                            results[snr]['ber'].append(ber)
                            results[snr]['affected_symbols'].append(
                                symbol_errors)

                            received_bytes = int(received_bits, 2).to_bytes(
                                rs_n,
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

                        output_text = fixed_coder.decompress_segments(
                            [received_bits_all]) if received_bits_all else ""
                        all_cleaned_inputs.append(cleaned_text)
                        all_output_texts.append(output_text)
                        results[snr]['total'] += 1
                        pbar.update(1)

            if all_cleaned_inputs and all_output_texts:
                print("Computing similarity...")
                print("Memory status before computing similarity:")
                memory_status()
                similarities = similarity.compute_similarity(all_cleaned_inputs,
                                                             all_output_texts)
                mean_similarity = np.mean(similarities) if similarities else 0.0
                print("Memory status after computing similarity:")
                memory_status()
            else:
                mean_similarity = 0.0

            results[snr][
                'similarity_scores'] = similarities if all_cleaned_inputs else []

            snr_end_time = time.time()
            snr_duration = snr_end_time - snr_start_time
            print(f"Time taken for SNR {snr} dB: {snr_duration:.2f} seconds")

            all_scores.append({
                'SNR': snr,
                'similarity_score': mean_similarity,
                'method': "5bit + rs",
                'total_samples': results[snr]['total'],
                'avg_ber': np.mean(results[snr]['ber']) if results[snr][
                    'ber'] else 0
            })

    finally:
        tqdm_module.tqdm.__init__ = original_tqdm_init

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    if all_scores:
        try:
            print("Collected scores:", all_scores)
            scores_df = pd.DataFrame(all_scores)
            scores_df.to_csv(scores_file, index=False)
            print(f"\nResults saved to: {eval_dir}")
            print(f"Scores file: {scores_file}")
            saved_df = pd.read_csv(scores_file)
            print(
                f"Contents of saved scores file:\n{saved_df.to_string(index=False)}")
        except Exception as e:
            print(f"Error saving evaluation scores to {scores_file}: {e}")

    for snr in snrs:
        avg_similarity = np.mean(results[snr]['similarity_scores']) if \
            results[snr]['similarity_scores'] else 0.0
        avg_ber = np.mean(results[snr]['ber']) if results[snr]['ber'] else 0
        avg_affected_symbols = np.mean(results[snr]['affected_symbols']) if \
            results[snr]['affected_symbols'] else 0
        print(
            f"SNR {snr} dB: Similarity={avg_similarity:.4f}, BER={avg_ber:.6f}, "
            f"AffectedSym={avg_affected_symbols:.2f}, Total Samples={results[snr]['total']}")

    try:
        eval_dir_contents = os.listdir(eval_dir)
        print(
            f"Contents of evaluation directory '{eval_dir}' after saving: {eval_dir_contents}")
    except Exception as e:
        print(
            f"Error accessing evaluation directory '{eval_dir}' after saving: {e}")
    print(f"Total time taken for batch test: {total_duration:.2f} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="5-bit + RS Test Script with Similarity")
    parser.add_argument('--channel', default='TimeVaryingRician', type=str,
                        help='Channel type (e.g., Rayleigh)')
    parser.add_argument('--MAX-LENGTH', default=30, type=int,
                        help='Maximum sentence length')
    parser.add_argument('--MIN-LENGTH', default=4, type=int,
                        help='Minimum sentence length')
    parser.add_argument('--d-model', default=128, type=int,
                        help='Model dimension (placeholder)')
    parser.add_argument('--num-layers', default=4, type=int,
                        help='Number of layers (placeholder)')
    parser.add_argument('--num-heads', default=8, type=int,
                        help='Number of attention heads (placeholder)')
    parser.add_argument('--dff', default=512, type=int,
                        help='Feed-forward dimension (placeholder)')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size')
    parser.add_argument('--epochs', default=1, type=int,
                        help='Number of epochs (single run here)')
    args = parser.parse_args()

    test_eur = EurDataset('test')
    snrs = [0]
    modulation = '64qam'
    rs_n = 255
    rs_k = 63

    batch_loader = sample_dataset_sentences(test_eur, num_samples=len(test_eur),
                                            batch_size=args.batch_size)
    batch_test_5bit_rs(batch_loader, snrs, modulation, args, rs_n, rs_k)
