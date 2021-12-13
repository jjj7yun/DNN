import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from utils import *
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW

class LSTM_classifier(nn.Module):
    def __init__(self, device, embed_dim, hidden_dim):
        super(LSTM_classifier, self).__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.tokenizer = AutoTokenizer.from_pretrained('monologg/koelectra-small-v3-discriminator')
        self.tokenizer.padding_side = 'left'

        self.build_model()

        self.to(device)
    
    def build_model(self):

        """
        self.word_embedding: (vocab_size, embed_dim)
        self.layer1: (hidden_dim, 1)
        """

        self.word_embedding = nn.Embedding(self.tokenizer.vocab_size, self.embed_dim) 
        self.RNN = nn.LSTM(self.embed_dim,self.hidden_dim,1,batch_first=True)
        self.layer1 = nn.Linear(self.hidden_dim,1)
        self.layer2 = nn.Linear(256,1)



        self.sigmoid = nn.Sigmoid()
        self.BCE_loss = nn.BCELoss()

    def forward(self, batch_text_ids):

        """
        batch_text_ids: (batch_size, max_length)
        output: (1, batch_size, 1)
        """


        output = self.word_embedding(batch_text_ids)
        output, hidden_dim = self.RNN(output)
        output = self.layer1(output)
        output = output.permute(0,2,1)
        output=self.layer2(output)


        output = self.sigmoid(output)

        return output.squeeze()

    def train_model(self, X_train, X_valid, y_train, y_valid, num_epochs, batch_size, learning_rate):

        self.optimizer = AdamW(self.parameters(), lr=learning_rate)
        y_train = np.array(y_train)

        loss_log = []
        for e in range(num_epochs):
            epoch_loss = 0
            batch_loader = DataBatcher(np.arange(len(X_train)), batch_size=batch_size)
            for b, batch_indices in enumerate(tqdm(batch_loader, desc=f'> {e+1} epoch training ...', dynamic_ncols=True)):
                self.optimizer.zero_grad()
                batch_text = [X_train[idx] for idx in batch_indices]
                

                id = self.tokenizer(batch_text, truncation = True, max_length =256, padding = 'max_length')
                batch_text_ids=torch.LongTensor(id.input_ids).to(self.device)
                out = self.forward(batch_text_ids)



                ###########################################################

                batch_labels = torch.Tensor(y_train[batch_indices]).to(self.device)
                loss = self.BCE_loss(out, batch_labels)
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            loss_log.append(epoch_loss)
            
            valid_accuracy, valid_loss = self.predict(X_valid, y_valid, batch_size)
            print(f'>> [Epoch {e+1}] Total epoch loss: {epoch_loss:.2f} / Valid accuracy: {100*valid_accuracy:.2f}% / Valid loss: {valid_loss:.4f}')
        
        return loss_log
  
    def predict(self, X, y, batch_size, return_preds=False):
        y = np.array(y)
        BCE_loss = nn.BCELoss(reduction='sum')
        preds = torch.zeros(len(X))
        total_loss = 0

        with torch.no_grad():
            batch_loader = DataBatcher(np.arange(len(X)), batch_size=batch_size)
            for _, batch_indices in enumerate(tqdm(batch_loader, desc=f'> Predicting ...', dynamic_ncols=True)):
                batch_text = [X[idx] for idx in batch_indices]

                id = self.tokenizer(batch_text, truncation = True, max_length =256, padding = 'max_length')
                batch_text_ids=torch.LongTensor(id.input_ids).to(self.device)
                out = self.forward(batch_text_ids)



                ###########################################################
                
                preds[batch_indices] = (out>0.5).float().cpu()
                batch_labels = torch.Tensor(y[batch_indices]).to(self.device)
                loss = BCE_loss(out, batch_labels)
                total_loss += loss

            accuracy = (preds.numpy() == y).sum() / y.shape[0]
            loss = total_loss / y.shape[0]
        
        if return_preds:
            return accuracy, loss, preds.numpy()
        else:
            return accuracy, loss