import torch
import numpy as np
import torch.nn as nn
from test import testing
from bilstm import BiLSTM
import torch.nn.functional as F


def train_model(opt, le, embedding_matrix, train_loader, valid_loader, X_cv, test_y):
    model = BiLSTM(le, opt, embedding_matrix)
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    train_loss = []
    accuracy_list = []

    for epoch in range(opt.num_train_epochs):

        avg_loss = 0.
        model.train()
        model.type = 'learning'
        for i, (x_batch, y_batch) in enumerate(train_loader):
            # Predict/Forward Pass
            y_pred = model(x_batch)
            # Compute loss
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)

        # Set model to validation configuration -Doesn't get trained here
        model.eval()
        avg_val_loss = 0.
        val_preds = np.zeros((len(X_cv), len(le.classes_)))

        for i, (x_batch, y_batch) in enumerate(valid_loader):
            y_pred = model(x_batch).detach()
            avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
            # keep/store predictions
            val_preds[i * opt.batch_size:(i + 1) * opt.batch_size] = F.softmax(y_pred,
                                                                               dim=1).cpu().numpy()  # Check

        # Check Accuracy
        train_loss.append(avg_loss)
        # valid_loss.append(avg_val_loss)
        print('Epoch {}/{} \t training loss={:.4f}  \t '.format(epoch + 1, opt.num_train_epochs, avg_loss), end='')
        accuracy_list = testing(model, X_cv, le, valid_loader, loss_fn, opt, test_y, accuracy_list)

    torch.save(model.state_dict(), f'../model/imdb/bilstm_model_imdb.pth')
    return model