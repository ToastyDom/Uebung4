from torch.utils.data.dataset import Dataset as TorchDataset
import csv
import torch
import os.path as op
from nltk.tokenize import wordpunct_tokenize
import json

def custom_collate_fn(batch):
    """
    HINT: If you want to use the torch.utils.data.DataLoader, you will have to change the default collation function.
    """
    bow = torch.LongTensor([item[0] for item in batch])
    target = torch.LongTensor([int(item[1]) for item in batch])
    return bow, target


def tokenize(text):
    text = text.lower() # lowercase
    tokens = wordpunct_tokenize(text) # tokenize
    return tokens


def build_bow_vector(token_sentence, token_dict):
    vector = [0] * len(token_dict)
    for token_idx in token_sentence:
        if token_idx in token_dict:
            vector[token_idx] += 1
    return vector



class Vocab():
    def __init__(self,datapath):
        
        
        if op.exists("./idx2token.json"):
            print("vocab exists already")
            with open('token2idx.json') as json_file:
                self.token2idx = json.load(json_file)
            print("first file loaded")
            with open('idx2token.json') as json_file:
                self.idx2token = json.load(json_file)
            print("second file loaded")
        else:
            print("Create Vocabulary")          
            data_train = op.join(datapath, "train.tsv")
            data_test = op.join(datapath, "test.tsv")
            data_dev = op.join(datapath, "dev.tsv")
            
            self.all_data = []
            
            print("Load all data from all files")
            for datafile in [data_train, data_test, data_dev]:
                with open(datafile) as file:
                    tsv_file = csv.reader(file, delimiter="\t")
                    for line in tsv_file:
                        self.all_data.append(line)
                        
            
            #print(self.all_data)
            print("create tokens")
            all_tokens = [tokenize(sentence[1]) for sentence in self.all_data]
            
            # print(all_tokens)
            print("create set")
            total_vocabulary = sorted(set(token for doc in all_tokens for token in doc))
            
            #print(total_vocabulary)
            
            print("create dictionaries")
            self.token2idx = {token: idx for idx, token in enumerate(total_vocabulary)}
            self.idx2token = {idx: token for token, idx in self.token2idx.items()}
            
            # Save as json:
            print("save jsons")
            with open('token2idx.json', 'w') as fp:
                json.dump(self.token2idx, fp)
            with open('idx2token.json', 'w') as fp:
                json.dump(self.idx2token, fp)
        
        
    def get_idx2token(self):
        return self.idx2token
    
    def get_token2idx(self):
        return self.token2idx
        


class Dataset(TorchDataset):
    def __init__(self, dataset, vocab_list):  # TODO: Implement
        """

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
        self.tokens = [tokenize(sentence) for sentence in self.sentences]
        
        # Create bag of words
        print("Create bag of words")
        self.bow = []
        for tok_sentence in self.tokens:
            sent_vec = []
            for token in vocab_list:
                if token in tok_sentence:
                    sent_vec.append(1)
                else:
                    sent_vec.append(0)
            self.bow.append(sent_vec)
                
                    
        
                
        # # Create indexes for each word
        # self.word_index = {}
        # for index, sentence in self.all_data:
        #     for word in sentence.split(" "):
        #         if word not in self.word_index:
        #             self.word_index[word] = len(self.word_index)
                    
                    
        # # Only get current dataset
        # current_dataset = op.join(datapath, dataset)
        # self.current_data = []

        # with open(current_dataset) as file:
        #     tsv_file = csv.reader(file, delimiter="\t")
        #     for line in tsv_file:
        #         self.current_data.append(line)
                    
        # # Make BOW-Vector for each sentence
        # self.bow_vectors = []
        # for index, sentence in self.current_data:
        #     vec = torch.zeros(len(self.word_index))  # the length of all available words
        #     for word in sentence.split(" "):
        #         vec[self.word_index[word]] +=1
        #     self.bow_vectors.append([float(index), vec])
        
            
        


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
