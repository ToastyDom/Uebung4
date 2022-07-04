from distutils import core
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch.nn as nn

from nlpds.data import Dataset, custom_collate_fn, Vocab
from nlpds.model import BagOfWordsClassifier

import numpy as np


from sklearn.metrics import classification_report

###############################################################
# NOTE: This is just an example outline of a training script. #
# You do not have to use any of this!                         #
###############################################################
if __name__ == '__main__':
    Vocabs = Vocab("data/data/uebung4", size=5000)
    dict_token_idx = Vocabs.get_dict_token_idx()
    dict_id_token = Vocabs.get_dict_id_token()
    train_dataset = Dataset("data/data/uebung4/train.tsv", dict_token_idx, dict_id_token)  # TODO
    dev_dataset = Dataset("data/data/uebung4/dev.tsv", dict_token_idx, dict_id_token)  # TODO
    test_dataset = Dataset("data/data/uebung4/test.tsv", dict_token_idx, dict_id_token)  # TODO
    
    
    # print(train_dataset.sentences[1])
    # print(train_dataset.targets[1])
    # print(train_dataset.tokens[1])
    # print(train_dataset.indexed_tokens[1])
    # print(train_dataset.bow[1])

    
    
    """Training settings"""
    num_epochs = 20  # TODO
    batch_size = 8  # TODO
    learning_rate = 0.5

    model = BagOfWordsClassifier(vocab_size=len(dict_token_idx), num_labels=5, hidden1=32,hidden2=128, hidden3=64, batchsize=batch_size, embedding_dim=300)  # TODO
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # TODO: torch.optim.?
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, 1)
    
    
    
    """Initiate Training"""
    print("Start Training:")
    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        
        """Training procedure"""
        total_loss, total = 0, 0
        model.train()
        for batch in DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate_fn):
            
            # Reset gradient
            optimizer.zero_grad()
            
            # Forward pass
            output = model(batch[0])
            
            # Compute loss
            loss = criterion(output, batch[1])
            
            # Perform gradient descent, backwards pass
            loss.backward()

            # Take a step in the right direction
            optimizer.step()
            scheduler.step()

            # Record metrics
            total_loss += loss.item()
            total += len(batch[1])
        print("Train:", total_loss / total)
        
        
        
        """Validation procedure"""
        total_loss, total = 0, 0
        model.eval() # change to evaluation mode
        with torch.no_grad():
            for batch_val in DataLoader(dev_dataset, batch_size=batch_size, collate_fn=custom_collate_fn):
                
                # Forward pass
                output = model(batch_val[0])

                # Calculate how wrong the model is
                loss = criterion(output, batch_val[1])

                # Record metrics
                total_loss += loss.item()
                total += len(batch_val[1])
        print("Val:", total_loss / total)
        
        
    
    
    
    """Initiate Testing"""
    print("Start testing")
    
    model.eval()
    test_accuracy, n_examples = 0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch_test in DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_collate_fn):
            
            inputs = batch_test[0]
            target = batch_test[1]
            prediction = model.predict(inputs)
     
            y_true.extend(target)
            y_pred.extend(prediction)
            
    print(classification_report(y_true, y_pred))
        
        
    
