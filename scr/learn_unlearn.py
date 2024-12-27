import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from test import testing


def get_learn_unlearn(le, model, opt, X_cv, test_y, retain_loader, forget_loader, valid_loader):
    accuracy_rank = {}
    for rank in range(opt.ranks):
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
        # model.cuda()
        model.rank = rank + 1
        accuracy_list = []
        learning_start_time = time.time()
        for turn in range(opt.n_turn):
            print(f'-----------------Turn={turn}-----------------')
            # Unlearning
            print('Unlearning')
            for epoch_unlr in range(opt.unlearning_epoch):
                start_time_unlearning = time.time()
                model.type = 'unlearning'
                model.epoch = epoch_unlr
                model.turn = turn
                avg_loss = 0
                for i, (x_batch, y_batch) in enumerate(forget_loader):
                    model.train()
                    # Predict/Forward Pass
                    y_pred = model(x_batch)
                    # Compute loss
                    loss = loss_fn(y_pred, y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    avg_loss += loss.item() / len(forget_loader)

                # Check Accuracy
                print('Epoch {}/{} \t training loss={:.4f} \t'.format(epoch_unlr + 1, opt.unlearning_epoch, avg_loss),
                      end='')
                accuracy_list = testing(model, X_cv, le, valid_loader, loss_fn, opt, test_y, accuracy_list)

            # Learning
            print('Learning')
            for epoch_lr in range(opt.learning_epoch):
                # Set model to train configuration
                model.train()
                model.type = 'learning'
                model.epoch = epoch_lr
                model.turn = turn
                avg_loss = 0.
                for i, (x_batch, y_batch) in enumerate(retain_loader):
                    # Predict/Forward Pass
                    y_pred = model(x_batch)
                    # Compute loss
                    loss = loss_fn(y_pred, y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    avg_loss += loss.item() / len(retain_loader)

                # Check Accuracy
                print('Epoch {}/{} \t training loss={:.4f} \t'.format(epoch_lr + 1, opt.learning_epoch, avg_loss),
                      end='')
                accuracy_list = testing(model, X_cv, le, valid_loader, loss_fn, opt, test_y, accuracy_list)

        learning_elapsed_time = time.time() - learning_start_time
        print('Learning time={:.2f}s'.format(learning_elapsed_time))

        accuracy_rank[rank] = accuracy_list
        print(f'.............Rank = {rank} done............')
    df = pd.DataFrame.from_dict(accuracy_rank)
    df.to_csv('../results/IMDB_Result.csv', index=False)
    return model
