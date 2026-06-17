import torch
import torch.nn as nn

class ReviewModel(nn.Module):
    def __init__(self,vocab_size,embedding_dim=128,hidden_dim=64,num_classes=2):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_dim,padding_idx=0)
        self.relu=nn.ReLU()
        self.lstm=nn.LSTM(input_size=embedding_dim,hidden_size=hidden_dim,batch_first=True) 
        self.drop=nn.Dropout(0.33)
        self.l2=nn.Linear(hidden_dim,num_classes)

    def forward(self,token_ids):
        embedded=self.embedding(token_ids)

        output,(h,c)=self.lstm(embedded)
        pooled=output.mean(dim=1)
        hidden=self.relu(pooled)
        hidden=self.drop(hidden)
        logits=self.l2(hidden)
        return logits