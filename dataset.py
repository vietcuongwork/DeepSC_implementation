# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author: Huiqiang Xie
@File: EurDataset.py
@Time: 2021/3/31 23:20
"""

import pickle  # Module for serializing and deserializing Python objects

from torch.utils.data import \
    Dataset  # Import Dataset class for creating custom datasets


class EurDataset(Dataset):
    """
    Custom dataset class for loading European language dataset.
    Inherits from PyTorch's Dataset class.
    """

    def __init__(self, split='train'):
        """
        Initializes the dataset by loading the preprocessed data from a pickle file.
        Args:
            split (str): Determines which dataset to load (train/test/validation).
        """
        data_dir = './data/'  # Directory where dataset files are stored

        # Load the dataset from the corresponding pickle file (e.g., train_data.pkl, test_data.pkl)
        with open(data_dir + '{}_data.pkl'.format(split), 'rb') as f:
            self.data = pickle.load(f)  # Deserialize the data

    def __getitem__(self, index):
        """
        Retrieves a single data sample (sentence) from the dataset.
        Args:
            index (int): The index of the sample.
        Returns:
            sents: The sentence at the given index.
        """
        sents = self.data[index]
        return sents

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.data)


import numpy as np
import torch


# def collate_data(batch):
#     """
#     Custom function to process a batch of sentences.
#     It pads sentences to the maximum length in the batch to ensure uniform tensor shape.
#     Args:
#         batch: A batch of tokenized sentences.
#     Returns:
#         torch.Tensor: A tensor containing the padded sentences.
#     """
#     batch_size = len(batch)  # Number of sentences in the batch
#
#     # Find the maximum sentence length in the batch
#     max_len = max(map(len, batch))
#
#     # Find the longest sentence
#     longest_sentence = max(batch, key=len)
#
#     # # Print the uniform length and the longest sentence
#     # print(f"\nUniform length for this batch: {max_len}")
#     # print(
#     #     f"Longest sentence: {longest_sentence} (Length: {len(longest_sentence)})")
#
#     # Initialize a NumPy array with zeros to hold the padded sentences
#     sents = np.zeros((batch_size, max_len), dtype=np.int64)
#
#     # Sort sentences by length in descending order (helps with packed sequences in NLP models)
#     sort_by_len = sorted(batch, key=lambda x: len(x), reverse=True)
#
#     # Copy sentences into the NumPy array, padding shorter sentences with zeros
#     for i, sent in enumerate(sort_by_len):
#         length = len(sent)
#         sents[i, :length] = sent  # Fill the row with the actual sentence tokens
#
#     return torch.from_numpy(
#         sents)  # Convert NumPy array to a PyTorch tensor for model input

# Force padding to 30
def collate_data(batch):
    batch_size = len(batch)
    target_len = 30  # Fixed length

    sents = np.zeros((batch_size, target_len),
                     dtype=np.int64)  # Always [128, 30]
    sort_by_len = sorted(batch, key=lambda x: len(x), reverse=True)

    for i, sent in enumerate(sort_by_len):
        length = min(len(sent), target_len)  # Truncate if longer than 30
        sents[i, :length] = sent[:length]  # Fill, rest stays 0

    # print(f"Batch padded to: {target_len}, Sample: {sents[0].tolist()}")
    return torch.from_numpy(sents)
