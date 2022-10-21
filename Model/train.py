import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch

def validation(model, test_loader, criterion, device):
    model.eval()
    from sklearn.metrics import f1_score, accuracy_score
    pred_proba_label = []
    true_label = []
    val_loss = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
    
            inputs = batch['mri'][tio.DATA].to(device) 
            label =  batch['label'].to(device)
     
            output = model(inputs)
            model_pred = torch.sigmoid(output).to('cpu')

            loss = criterion(output, label)
            pred_proba_label += model_pred.tolist()
            true_label += label.to('cpu').tolist()
            
    pred_label = np.where(np.array(pred_proba_label)>0.5, 1, 0)
    f1 = f1_score(true_label, pred_label, average='macro')
    acc = accuracy_score(true_label, pred_label)
    print("f1: " ,f1, "Acc : ", acc)


def train(model, Epochs,train_loader, test_loader, device):
    #device = 'cuda'
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    
    citerition = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = 1e-4)

    for epoch in range(1, Epochs):
        model.train()
        train_loss = []

        for batch_idx, batch in enumerate(tqdm(train_loader)):
           
            optimizer.zero_grad()
            inputs = batch['mri'][tio.DATA].to(device) 
            label =  batch['label'].to(device)
         
            output = model(inputs)
            loss = citerition(output, label)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
        
        if epoch % 10 ==1 :
            print("EPOCH : ",epoch)
            print("Train LOSS : ", np.mean(np.array(train_loss)))
            validation(model, test_loader, criterion=citerition, device=device)
            ## SAVE 
            torch.save(model, f"./{epoch}_epoch.pt")
