import torch
import torch.nn as nn
import numpy as np


class BagOfWordsClassifier(nn.Module):
    def __init__(self, vocab_size, num_labels, hidden1, hidden2, hidden3, batchsize, embedding_dim,):  # TODO: Implement
        """
        TODO: Write a description for this class.
        HINT: This should give you good idea on how to build your model,
              but you can change anything here at will.
              
        The model consists out of four fully connected layers.
        The input has to be the size of the vocab, because no matter what the vocab is, it is always the same size and this is
        going to b the input of the model
        
        The next fully connected layers have to feed into each other so they depend on the layers after
        
        The final layer ends with the num_labels because that is what we want to classify
        """
        super(BagOfWordsClassifier, self).__init__()
        self.batchsize = batchsize
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, num_labels)

    def forward(self, input: torch.LongTensor) -> torch.FloatTensor:
        """
        A model forward pass.

        Calculates sentence representations by calculating the embedding input word (indices)
        and pooling all sentence word embeddings for a single fixed-sized representation,
        that is then fed to the classifier.

        :param input: A 2-dimensional tensor that contains indices of the input words.
        :return: A 2-dimensional tensor with the 'logits' output of the classification layer.
        """

        x = self.embedding(input)
        x = x.mean(dim=1)

        x = self.fc1(x)
        x = nn.functional.relu(x)
        
        x = self.fc2(x)
        x = nn.functional.relu(x)
        
        x = self.fc3(x)
        x = nn.functional.relu(x)
        
        x = self.fc4(x)
        x = torch.sigmoid(x)
        
        return x
        
        

    def predict(self, input: torch.LongTensor) -> torch.LongTensor:
        """
        Similar to BaseModule.forward, but returns the predicted classes instead of raw logits.

        :param input: A 2-dimensional tensor that contains indices of the input words.
        :return: A 1-dimensional tensor with the predicted classes.
        """
        
        x = self.embedding(input)
        x = x.mean(dim=1)

        x = self.fc1(x)
        x = nn.functional.relu(x)
        
        x = self.fc2(x)
        x = nn.functional.relu(x)
        
        x = self.fc3(x)
        x = nn.functional.relu(x)
        
        x = self.fc4(x)
        x = torch.sigmoid(x)
        
        final_prediction = np.argmax(x, axis=1)
        
        return final_prediction
