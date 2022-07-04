from nltk.corpus import stopwords
from torch.utils.data.dataset import Dataset as TorchDataset
import csv
import torch
import os.path as op
from nltk.tokenize import wordpunct_tokenize
import json
from collections import Counter
import re
import nltk
import numpy as np


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def custom_collate_fn(batch):
    """
    HINT: If you want to use the torch.utils.data.DataLoader, you will have to change the default collation function.
    """
    bow = torch.LongTensor([item[0] for item in batch])
    target = torch.LongTensor([int(item[1]) for item in batch])
    return bow, target


def tokenize(text):
    """This function tokenizes and cleans sentences

    Args:
        text (list[str]): sentence with words

    Returns:
        tokens: list of cleaned tokens
    """
    text_without_special = re.sub(r'[^\w\s]', '', text)  # remove special characters
    text_lower = text_without_special.lower()  # lowercase
    tokens = wordpunct_tokenize(text_lower)  # tokenize
    tokens_no_stopwords = [token for token in tokens if token not in stop_words] # remove stopwords
    clean_tokens = [re.sub(r'[0-9]+', '<NUM>', token) for token in tokens_no_stopwords] # replace numbers

    return clean_tokens

def tokenize_data(text, vocab):
    """This function tokenizes and cleans sentences for the DataLoaders. It addiitionally checks if the words are available
    in the vocabulary

    Args:
        text (list[str]): sentence with words

    Returns:
        tokens: list of cleaned tokens
    """
    text_without_special = re.sub(r'[^\w\s]', '', text)  # remove special characters
    text_lower = text_without_special.lower()  # lowercase
    tokens = wordpunct_tokenize(text_lower)  # tokenize
    tokens_no_stopworkds = [token for token in tokens if token not in stop_words]
    clean_tokens = [re.sub(r'[0-9]+', '<NUM>', token) for token in tokens_no_stopworkds] # replace numbers
    clean_tokens = [token for token in clean_tokens if token in vocab] # only the tokens that are in our vocabulary
    return clean_tokens


def build_bow_vector(token_sentence, token_dict):
    """Creates a bow_vector from the vocabulary

    Args:
        token_sentence (list of str): sentence to turn into bow
        token_dict (dict): vocabulary

    Returns:
        vector: bow vector
    """
    vector = np.zeros(len(token_dict))
    for token_idx in token_sentence:
        if token_idx in token_dict:
            vector[token_idx] += 1
    return vector


class Vocab():
    def __init__(self, datapath, size):
        """This class creates a vocabulary from all training/testing data to be used for creating the bow-vectors

        Args:
            datapath (str): path to where the data is located
            size(int): Size of vocabulary
        """

        # If vocabulary already exsists load it
        if op.exists("./dict_id_token.json"):
            print("vocab exists already")
            
            with open('dict_token_idx.json') as json_file:
                self.dict_token_idx = json.load(json_file)
                print("first file loaded")
                
            with open('dict_id_token.json') as json_file:
                self.dict_id_token = json.load(json_file)
                print("second file loaded")
                
        # Else create new
        else:
            print("Create Vocabulary")
            data_train = op.join(datapath, "train.tsv")
            data_test = op.join(datapath, "test.tsv")
            data_dev = op.join(datapath, "dev.tsv")

            self.all_data = []

            # Load data from all files
            print("Load all data from all files")
            for datafile in [data_train, data_test, data_dev]:
                with open(datafile) as file:
                    tsv_file = csv.reader(file, delimiter="\t")
                    for line in tsv_file:
                        self.all_data.append(line[1])

            # Create a list of tokens from all those words
            print("create tokens")
            all_tokens = [tokenize(sentence) for sentence in self.all_data]
            all_tokens = [item for sublist in all_tokens for item in sublist]


            # Sort by occurence, remove duplicates, take the firs #size of the entire vocab
            # Less frequent words we dont really need
            print("create_common")
            common_tokens = sorted(
                set(list(zip(*Counter(all_tokens).most_common(size)))[0]))


            # Create dictionaries from the list of common tokens
            print("create dictionaries")
            self.dict_token_idx = {token: idx for idx,
                              token in enumerate(common_tokens)}
            self.dict_id_token = {idx: token for token,
                              idx in self.dict_token_idx.items()}

            # Save as json:
            print("save jsons")
            with open('dict_token_idx.json', 'w') as fp:
                json.dump(self.dict_token_idx, fp)
            with open('dict_id_token.json', 'w') as fp:
                json.dump(self.dict_id_token, fp)

    def get_dict_id_token(self):
        """Returns dictionary"""
        return self.dict_id_token

    def get_dict_token_idx(self):
        """Returns dictionary"""
        return self.dict_token_idx


class Dataset(TorchDataset):
    def __init__(self, dataset, dict_token_idx, dict_id_token):  # TODO: Implement
        """This class prepares the data for the Dataloader

        Args:
            dataset (str): Path to the dataset to be loaded
            dict_token_idx (dict): Vocab Dictionary
            dict_id_token (dict): Vocab Dictionary with index as indice
        """
        super(Dataset, self).__init__()
        
        
        # Colllect data from file
        datafile = op.join(dataset)
        
        
        # Save data in target and sentence list
        print("load dataset:", dataset)
        self.sentences = []
        self.targets = []
        with open(datafile) as file:
            tsv_file = csv.reader(file, delimiter="\t")
            for line in tsv_file:
                self.targets.append(line[0])
                self.sentences.append(line[1])

               
        # Convert sentences into tokens 
        print("Create Tokens")
        self.tokens = [tokenize_data(sentence, dict_token_idx) for sentence in self.sentences]

        # Instead of tokens create list of indexes
        print("Create indexed Tokens")
        self.indexed_tokens = []
        for sentence in self.tokens:
          sub_indexed = [dict_token_idx[token] for token in sentence]
          self.indexed_tokens.append(sub_indexed)
        
        # Create the bag of words
        print("Create bag of words")
        self.bow = []
        for tok_sentence in self.indexed_tokens:
            self.bow.append(build_bow_vector(tok_sentence, dict_id_token))       


    def __len__(self):
        """
        Returns the length of this dataset.

        :return: The length of this dataset.
        """
        return len(self.targets)

    def __getitem__(self, item):
        """
        Returns the item (the features and corresponding label) at the given index.

        :param item: The index of the item to get.
        :return: The item at the corresponding index.
        """
        return (self.bow[item], self.targets[item])
