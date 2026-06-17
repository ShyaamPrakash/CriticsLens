import torch
import torch.nn as nn
import joblib
import pandas as pd
from model import ReviewModel
from data import load_and_split_data, build_vocab_from_texts, make_dataloader
from evaluate import evaluate, plot_losses


class EarlyStopper():
    def __init__(self,model,patience=8):
        self.patience_counter=0
        self.max_patience=patience
        self.model=model
        self.best_val_loss=float('inf')
        self.best_weights=None
        self.train_losses=[]
        self.val_losses=[]


    def trainOneEpoch(self,train_loader,criterion,optimizer):
        self.model.train()
        epochloss=0.0
        for X,y in train_loader:
            optimizer.zero_grad()
            output=self.model(X)
            loss=criterion(output,y)
            loss.backward()
            optimizer.step()
            epochloss+=loss.item()
        return epochloss/len(train_loader)
    

    def validator(self,val_loader,criterion):
        self.model.eval()
        valloss=0.0
        with torch.no_grad():

            for X,y in val_loader:
                output=self.model(X)
                loss=criterion(output,y)
                valloss+=loss.item()

        return valloss/len(val_loader)


    def fit(self,train_loader,val_loader,criterion,optimizer,epochs=100):
        for epoch in range(epochs):

            train_loss=self.trainOneEpoch(train_loader=train_loader,criterion=criterion,optimizer=optimizer)
            val_loss=self.validator(val_loader=val_loader,criterion=criterion)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            print(epoch+1)
            if val_loss<self.best_val_loss:
                self.best_val_loss=val_loss
                self.patience_counter=0
                self.best_weights={k:v.clone() for k,v in self.model.state_dict().items()}
            else:
                self.patience_counter+=1
            if self.patience_counter>=self.max_patience:
                print("early stopping at epoch",epoch)

                break

            if(epoch+1)%10==0:
                print(f"Epoch {epoch+1}\nTrain Loss:{train_loss}\nVal Loss:{val_loss}\nPatience Counter:{self.patience_counter}/{self.max_patience}")
        

        self.model.load_state_dict(self.best_weights)

        return self.train_losses,self.val_losses



def main():
    (train_t, val_t, test_t), (train_l, val_l, test_l) = load_and_split_data('IMDB Dataset.csv')

    vocab = build_vocab_from_texts(train_t, min_freq=5)

    train_dl = make_dataloader(train_t, train_l, vocab, batch_size=32, shuffle=True)
    val_dl = make_dataloader(val_t, val_l, vocab, batch_size=32, shuffle=False)
    test_dl = make_dataloader(test_t, test_l, vocab, batch_size=32, shuffle=False)

    model = ReviewModel(vocab_size=len(vocab), embedding_dim=128, hidden_dim=64, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    trainer = EarlyStopper(model, patience=8)
    train_losses, val_losses = trainer.fit(train_dl, val_dl, criterion, optimizer, epochs=100)

    joblib.dump(model.state_dict(), 'sentiment_model.pkl')
    joblib.dump(vocab, 'vocab.pkl')

    test_acc = evaluate(model, test_dl)
    print(f"Test Accuracy: {test_acc:.4f}")

    plot_losses(train_losses, val_losses)


if __name__ == '__main__':
    main()