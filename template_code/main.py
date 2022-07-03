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
    vocab_list = Vocab("data/data/uebung4").get_token2idx()
    train_dataset = Dataset("data/data/uebung4/train.tsv", vocab_list)  # TODO
    dev_dataset = Dataset("data/data/uebung4/dev.tsv", vocab_list)  # TODO
    test_dataset = Dataset("data/data/uebung4/test.tsv", vocab_list)  # TODO
    
    
    
    num_epochs = 20  # TODO
    batch_size = 528  # TODO
    learning_rate = 5e-4

    model = BagOfWordsClassifier(vocab_size=len(vocab_list), num_labels=5, hidden1=100, hidden2=50, batchsize=batch_size, embedding_dim=300)  # TODO
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # TODO: torch.optim.?
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, 1)
    
    print("about to train")
    
    
    for epoch in range(num_epochs):
        print("Epoch:", epoch)
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
        
        
        
        # Now for validation
        model.eval()
        total_loss, total = 0, 0
        for batch_val in DataLoader(dev_dataset, batch_size=batch_size, collate_fn=custom_collate_fn):
            # Forward pass
            output = model(batch_val[0])

            # Calculate how wrong the model is
            loss = criterion(output, batch_val[1])

            # Record metrics
            total_loss += loss.item()
            total += len(batch_val[1])
        print("Val:", total_loss / total)
        
        
    
    
    print("testing!")
    
    model.eval()
    test_accuracy, n_examples = 0, 0
    y_true, y_pred = [], []
    input_type = 'bow'

    with torch.no_grad():
        for batch_test in DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_collate_fn):
            inputs = batch_test[0]
            target = batch_test[1]
            probs = model(inputs)
     
            probs = probs.detach().cpu().numpy()
            predictions = np.argmax(probs, axis=1)
            target = target.cpu().numpy()
            
            y_true.extend(predictions)
            y_pred.extend(target)
            
    print(classification_report(y_true, y_pred))
        
        
    

        # return total_loss / total
    
    # vocab = len(train_dataset.word_index)
    
    
    # print("vocab:", vocab)
    # print("vocab test:", len(test_dataset.word_index) )
    
    # num_epochs = 20  # TODO
    # batch_size = 8  # TODO
    # learning_rate = 0.05

    # model = BagOfWordsClassifier(vocab_size=vocab, num_labels=5, hidden1=100, hidden2=50, batchsize=batch_size)  # TODO

    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # TODO: torch.optim.?
    # criterion = nn.CrossEntropyLoss()
    # scheduler = CosineAnnealingLR(optimizer, 1)
    
    # device = torch.device(
    #     "cuda" if torch.cuda.is_available() else "cpu")  # TODO

    # total_loss, total = 0, 0

    # model.train()
    # for epoch in range(num_epochs):
    #     for batch in DataLoader(train_dataset, batch_size=batch_size):#, collate_fn=custom_collate_fn):
            
    #         inputs = batch[1]
    #         targets = torch.tensor(batch[0], dtype=torch.long)
            
    #         output = model(inputs)
     
    #         loss = criterion(output, targets)
            
    #         loss.backward()
            
    #         optimizer.step()
    #         scheduler.step()
            
    #         # Record metrics
    #         total_loss += loss.item()
    #         total += len(targets)
            
        
    #     print("Epoch:", epoch, "Loss:", total_loss / total)
            
            

    # model.eval()
    # with torch.no_grad():
    #     for batch in DataLoader(test_dataset, batch_size=batch_size):
    #         inputs = batch[1]
    #         targets = torch.tensor(batch[0], dtype=torch.long)
            
    #         output = model(inputs)
            
    #         loss = criterion(output, targets)
            
    #         # Record metrics
    #         total_loss += loss.item()
    #         total += len(targets)
    # print("Loss Testing:", total_loss / total)
