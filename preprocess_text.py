# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author: Huiqiang Xie
@File: text_preprocess.py
@Time: 2021/3/31 22:14
"""
import sys
from collections import Counter

import nltk
from matplotlib import pyplot as plt

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:44:08 2020

@author: hx301
"""
import argparse
import json
import os
import pickle
import re

import unicodedata
from tqdm import tqdm
from w3lib.html import remove_tags

# Download NLTK data (run once if not already installed)
nltk.download('punkt', quiet=True)
# Argument parser for handling input and output directories for text data processing
parser = argparse.ArgumentParser()
# parser.add_argument('--input-data-dir', default='europarl/txt/en', type=str)
parser.add_argument('--output-train-dir', default='train_data.pkl',
                    type=str)
parser.add_argument('--output-test-dir', default='test_data.pkl',
                    type=str)
parser.add_argument('--output-vocab', default='vocab.json', type=str)

# Special tokens used in the semantic communication model
SPECIAL_TOKENS = {
    '<PAD>': 0,
    '<START>': 1,
    '<END>': 2,
    '<UNK>': 3,
}


# Function to normalize unicode characters into ASCII
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


# Function to remove XML tags
def remove_tags(s):
    return re.sub(r'<[^>]+>', '', s)


# Function to remove annotations like (Protests), (FR), etc.
def remove_annotations(s):
    return re.sub(r'\s*\([^)]*\)', '', s)


# Function to extract spoken content, remove tags and metadata
def parse_and_extract_text(raw_text):
    text = remove_tags(raw_text)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    spoken_lines = []
    for line in lines:
        if not re.match(r'^\d+\.\s|^-\s|^rapporteur\.', line):
            spoken_lines.append(line)
    return '\n'.join(spoken_lines)


# Function to clean and preprocess text, matching DeepSC's punctuation behavior
def normalize_string(s):
    s = unicode_to_ascii(s)
    s = remove_annotations(s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    s = re.sub(r'([!.?])', r' \1', s)  # DeepSC behavior
    s = re.sub(r'\s+', r' ', s)
    s = s.lower().strip()
    return s


# Function to filter sentences by length, matching DeepSC's exact logic
def cutted_data(sentences, MIN_LENGTH=4, MAX_LENGTH=29):
    cutted_lines = []
    for line in sentences:
        length = len(line.split())
        if length > MIN_LENGTH and length < MAX_LENGTH:
            cutted_lines.append(line)
    return cutted_lines


# Optimized function to process and clean raw text from a file
def process(text_path):
    try:
        with open(text_path, 'r', encoding='utf-8') as fop:
            raw_data = fop.read()
    except FileNotFoundError:
        print(f"Error: File '{text_path}' not found.")
        sys.exit(1)

    # Extract spoken content first
    spoken_text = parse_and_extract_text(raw_data)

    # Pre-clean the entire text block
    cleaned_text = normalize_string(spoken_text)

    # Tokenize into sentences using NLTK
    sentences = nltk.sent_tokenize(cleaned_text)

    # Filter by length
    raw_data_input = cutted_data(sentences)

    return sentences, raw_data_input


# Debug function to process and print sentences at each stage
def debug_process(text_path):
    try:
        with open(text_path, 'r', encoding='utf-8') as fop:
            raw_data = fop.read()
    except FileNotFoundError:
        print(f"Error: File '{text_path}' not found.")
        sys.exit(1)

    print("\n### Raw Data (First 500 characters):")
    print(raw_data[:500], "...")

    # Extract spoken content first
    spoken_text = parse_and_extract_text(raw_data)
    print("\n### After Parsing (First 500 characters):")
    print(spoken_text[:500], "...")

    # Pre-clean the entire text block
    cleaned_text = normalize_string(spoken_text)
    print("\n### Normalized Text (First 500 characters):")
    print(cleaned_text[:500], "...")

    # Tokenize into sentences using NLTK
    sentences = nltk.sent_tokenize(cleaned_text)
    print(f"\n### Tokenized Sentences ({len(sentences)} sentences):")
    for i, sentence in enumerate(sentences, 1):
        print(f"{i}: {sentence}")

    # Filter by length
    filtered_sentences = cutted_data(sentences)
    print(f"\n### Filtered Sentences ({len(filtered_sentences)} sentences):")
    for i, sentence in enumerate(filtered_sentences, 1):
        print(f"{i}: {sentence}")

    return filtered_sentences


# Function to save preprocessed sentences as pickle files for later use
def save_clean_sentences(sentence, save_path):
    pickle.dump(sentence, open(save_path, 'wb'))
    print('Saved: %s' % save_path)


# Tokenizer that splits text into individual tokens (words) with special tokens for start and end
def tokenize(s, delim=' ', add_start_token=True, add_end_token=True,
             punct_to_keep=None, punct_to_remove=None):
    """
    Tokenize a sequence, converting a string s into a list of (string) tokens by
    splitting on the specified delimiter. Optionally keep or remove certain
    punctuation marks and add start and end tokens.
    """
    if punct_to_keep is not None:
        for p in punct_to_keep:
            s = s.replace(p, '%s%s' % (delim, p))

    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, '')

    tokens = s.split(delim)
    if add_start_token:
        tokens.insert(0, '<START>')
    if add_end_token:
        tokens.append('<END>')
    return tokens


# Build a vocabulary by counting token occurrences in the dataset
def build_vocab(sequences, token_to_idx={}, min_token_count=1, delim=' ',
                punct_to_keep=None, punct_to_remove=None, ):
    token_to_count = {}

    # Count frequency of each token in the dataset
    for seq in sequences:
        seq_tokens = tokenize(seq, delim=delim, punct_to_keep=punct_to_keep,
                              punct_to_remove=punct_to_remove,
                              add_start_token=False, add_end_token=False)
        for token in seq_tokens:
            if token not in token_to_count:
                token_to_count[token] = 0
            token_to_count[token] += 1

    # Include only tokens that meet the minimum frequency requirement
    for token, count in sorted(token_to_count.items()):
        if count >= min_token_count:
            token_to_idx[token] = len(token_to_idx)

    return token_to_idx


# Encode text sequences into numerical representations based on vocabulary
def encode(seq_tokens, token_to_idx, allow_unk=False):
    seq_idx = []
    for token in seq_tokens:
        if token not in token_to_idx:
            if allow_unk:
                token = '<UNK>'
            else:
                raise KeyError('Token "%s" not in vocab' % token)
        seq_idx.append(token_to_idx[token])
    return seq_idx


# Decode numerical representations back into tokens
def decode(seq_idx, idx_to_token, delim=None, stop_at_end=True):
    tokens = []
    for idx in seq_idx:
        tokens.append(idx_to_token[idx])
        if stop_at_end and tokens[-1] == '<END>':
            break
    if delim is None:
        return tokens
    else:
        return delim.join(tokens)


def explore_data(sentences, vocab):
    sentence_lengths = [len(s.split()) for s in sentences]
    most_common_words = Counter(" ".join(sentences).split()).most_common(20)

    print(f"Total sentences: {len(sentences)}")
    print(
        f"Avg sentence length: {sum(sentence_lengths) / len(sentence_lengths):.2f}")
    print(f"Max sentence length: {max(sentence_lengths)}")
    print(f"Min sentence length: {min(sentence_lengths)}")
    print("Most common words:")
    for word, freq in most_common_words:
        print(f"{word}: {freq}")

    plt.hist(sentence_lengths, bins=20, edgecolor='black')
    plt.xlabel("Sentence Length")
    plt.ylabel("Frequency")
    plt.title("Sentence Length Distribution")
    plt.show()


# Main function for preprocessing and saving train/test data
def main(args):
    total_sentences = []
    total_filtered_sentences = []

    data_dir = './data/'

    # Define multiple input directories
    args.input_data_dir = [os.path.join(data_dir, "europarl/txt/en")]

    args.output_train_dir = os.path.join(data_dir, args.output_train_dir)
    args.output_test_dir = os.path.join(data_dir, args.output_test_dir)
    args.output_vocab = os.path.join(data_dir, args.output_vocab)

    print("Input directories:", args.input_data_dir)

    print('Preprocess Raw Text')

    # Process and clean all text files in the input directories
    for input_dir in args.input_data_dir:
        if not os.path.exists(input_dir):
            print(f"Warning: Directory {input_dir} does not exist, skipping...")
            continue
        for fn in tqdm(os.listdir(input_dir)):
            if not fn.endswith('.txt'):
                continue
            file_path = os.path.join(input_dir, fn)

            sentences, filtered_sentences = process(file_path)
            total_sentences.extend(sentences)
            total_filtered_sentences.extend(filtered_sentences)

    # **Print total number of sentences before any processing**
    print(f"Total sentences before any processing: {len(total_sentences)}")

    # Compute statistics
    total_before_length_filter = len(total_sentences)
    total_after_length_filter = len(total_filtered_sentences)
    removed_by_length = total_before_length_filter - total_after_length_filter

    unique_sentences = list(set(total_filtered_sentences))
    removed_by_uniqueness = total_after_length_filter - len(unique_sentences)

    print("Total sentences removed by length filtering:", removed_by_length)
    print("Total sentences removed after removing duplicates:",
          removed_by_uniqueness)
    print("Number of sentences after filtering:", len(unique_sentences))

    print('Build Vocab')
    # Build the vocabulary from the cleaned dataset
    token_to_idx = build_vocab(
        unique_sentences, SPECIAL_TOKENS,
        punct_to_keep=[';', ','], punct_to_remove=['?', '.']
    )

    vocab = {'token_to_idx': token_to_idx}
    print('Number of words in Vocab:', len(token_to_idx))

    # Save the vocabulary as a JSON file
    if args.output_vocab:
        with open(args.output_vocab, 'w') as f:
            json.dump(vocab, f)

    print('Start encoding text')
    results = []
    # Encode each sentence into token indices
    for seq in tqdm(unique_sentences):
        words = tokenize(seq, punct_to_keep=[';', ','],
                         punct_to_remove=['?', '.'])
        tokens = [token_to_idx[word] for word in words]
        results.append(tokens)

    print('Writing Data')
    # Split the data into train and test sets
    train_data = results[: round(len(results) * 0.9)]
    test_data = results[round(len(results) * 0.9):]

    # **Print sample of train and test data**
    print("Sample from train data:",
          train_data[:5])  # Print first 5 samples from train data
    print("Sample from test data:",
          test_data[:5])  # Print first 5 samples from test data

    explore_data(unique_sentences, vocab)

    # Save the train and test data as pickle files
    with open(args.output_train_dir, 'wb') as f:
        pickle.dump(train_data, f)
    with open(args.output_test_dir, 'wb') as f:
        pickle.dump(test_data, f)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
    # debug_process('data/europarl/txt/en/ep-07-05-23-005-04.txt')
