import torch
import pandas as pd 

def train_loop(model, train_loader, loss_fn, optimizer, device):
    model.train()
    for X_, y_ in train_loader:
        X_ = X_.to(device)
        y_ = y_.to(device)
        pred = model(X_)
        loss = loss_fn(pred, y_)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def eval_loop(model, test_loader, loss_fn, device):
    model.eval()
    all_pred = []
    all_y = []
    with torch.no_grad():
        for X_, y_ in test_loader:
            X_ = X_.to(device)
            y_ = y_.to(device)
            pred = model(X_)
            all_pred.append(pred)
            all_y.append(y_)
        pred = torch.cat(all_pred)
        y_ = torch.cat(all_y)
        loss = loss_fn(pred, y_)
        acc = 100*((torch.argmax(pred, dim=1) == y_)*1.).mean().cpu()        
    return loss.cpu(), acc.cpu()


def evaluate_model(model, ds_in, ds_out, device):
    model = model.cpu()
    data_in = ds_in.values.transpose(1,0,2,3)
    data_out = ds_out.values
    X = torch.tensor(data_in).type(torch.float32)
    pred = model(X)
    df_out = pd.DataFrame(dict(truth=data_out,
                               pred = torch.argmax(pred, dim=1).cpu().detach().numpy(),
                               certainty = 100*torch.softmax(pred, dim=1).cpu().detach().numpy().max(axis=1)),
                          index=ds_out.time)
    model = model.to(device)
    return df_out
    