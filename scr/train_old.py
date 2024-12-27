import time
import torch
from tqdm.auto import tqdm
import pandas as pd

tqdm.pandas(desc='Progress')
# cross validation and metrics
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
from art.attacks.evasion import FastGradientMethod
from bilstm import BiLSTM
from prepare_data import get_data_loader
from test import testing


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
    # New param
    parser.add_argument("--embed_size", type=int, default=300)
    parser.add_argument("--max_features", type=int, default=120000)
    parser.add_argument("--maxlen", type=int, default=750)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--SEED", type=int, default=10)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--learning_epoch", type=int, default=3)
    parser.add_argument("--unlearning_epoch", type=int, default=5)
    parser.add_argument("--n_turn", type=int, default=5)
    parser.add_argument("--ranks", type=int, default=4)

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_epoches", type=int, default=5)

    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--word_hidden_size", type=int, default=50)
    parser.add_argument("--sent_hidden_size", type=int, default=50)
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=5,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--train_set", type=str, default="data/train.csv")
    parser.add_argument("--test_set", type=str, default="data/test.csv")
    parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--word2vec_path", type=str, default="data/glove.6B.50d.txt")
    parser.add_argument("--log_path", type=str, default="tensorboard/han_voc")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    args = parser.parse_args()
    return args





def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    accuracy_rank = {}

    train_loader, valid_loader, retain_loader, forget_loader, embedding_matrix, le, x_cv, test_y, X_forget, y_forget = get_data_loader(opt)

    for rank in range(opt.ranks):
        model = BiLSTM(le, opt, embedding_matrix)
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
        model.cuda()
        model.rank = rank + 1
        accuracy_list = []
        train_loss = []
        for turn in range(opt.n_turn):
            print(f'-----------------Turn={turn}-----------------')
            # Learning
            print('Learning')
            for epoch_lr in range(opt.learning_epoch):
                start_time = time.time()
                # Set model to train configuration
                model.train()
                model.type = 'learning'
                model.epoch = epoch_lr
                model.turn = turn

                avg_loss = 0.
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
                val_preds = np.zeros((len(x_cv), len(le.classes_)))

                for i, (x_batch, y_batch) in enumerate(valid_loader):
                    y_pred = model(x_batch).detach()
                    avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
                    # keep/store predictions
                    val_preds[i * opt.batch_size:(i + 1) * opt.batch_size] = F.softmax(y_pred,
                                                                                       dim=1).cpu().numpy()  # Check

                # Check Accuracy
                train_loss.append(avg_loss)
                # valid_loss.append(avg_val_loss)
                elapsed_time = time.time() - start_time
                print('Epoch {}/{} \t training loss={:.4f}  \t time={:.2f}s'.format(epoch_lr + 1, opt.learning_epoch,
                                                                                    avg_loss, elapsed_time), end='')

                accuracy_list = testing(model, x_cv, le, valid_loader, loss_fn, opt, test_y, accuracy_list)

            # Unlearning
            print('Unlearning')
            for epoch_unlr in range(opt.unlearning_epoch):
                start_time_unlearning = time.time()
                model.type = 'unlearning'
                model.epoch = epoch_unlr
                model.turn = turn

                avg_loss = 0.
                for i, (x_batch, y_batch) in enumerate(train_loader):
                    model.train()
                    # Predict/Forward Pass
                    y_pred = model(x_batch)
                    # Compute loss
                    loss = loss_fn(y_pred, y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    avg_loss += loss.item() / len(train_loader)
                    #
                    # Set model to validation configuration -Doesn't get trained here
                    model.eval()
                    avg_val_loss = 0.
                    val_preds = np.zeros((len(x_cv), len(le.classes_)))

                    for i, (x_batch, y_batch) in enumerate(valid_loader):
                        y_pred = model(x_batch).detach()
                        avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
                        # keep/store predictions
                        val_preds[i * opt.batch_size:(i + 1) * opt.batch_size] = F.softmax(y_pred, dim=1).cpu().numpy()

                    # Check Accuracy
                    # val_accuracy = sum(val_preds.argmax(axis=1) == test_y) / len(test_y)
                    train_loss.append(avg_loss)
                    # valid_loss.append(avg_val_loss)

                    # Step 6: Generate adversarial test examples
                    attack = FastGradientMethod(estimator=model, eps=0.2)
                    x_test_adv = attack.generate(x=X_forget)

                    # Step 7: Evaluate the ART classifier on adversarial test examples
                    predictions = model.predict(x_test_adv)
                    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_forget, axis=1)) / len(y_forget)
                    # mia_score_list.append(accuracy)
                    print("Accuracy on adversarial test examples: {:0.2f}%".format(accuracy * 100))






                elapsed_time = time.time() - start_time
                print(
                    'Epoch {}/{} \t training loss={:.4f}  \t time={:.2f}s'.format(epoch_unlr + 1, opt.unlearning_epoch,
                                                                                  avg_loss, elapsed_time), end='')
                accuracy_list = testing(model, x_cv, le, valid_loader, loss_fn, opt, test_y, accuracy_list)

                torch.save(model.state_dict(), f'../model/imdb/bilstm_model_rank_{rank}_turn{turn}.pth')
        # print(accuracy_list)
        accuracy_rank[rank] = accuracy_list
        print(f'.............Rank = {rank} done............')

    df = pd.DataFrame.from_dict(accuracy_rank)
    df.to_csv('../results/IMDB_Result.csv', index=False)
    # plt.plot(accuracy_list)
    # # Add labels and title (optional)
    # plt.xlabel('Turn')
    # plt.ylabel('Accuracy')
    # plt.title('Learning-Unlearning plot')
    #
    # # Show the plot
    # plt.show()
    #
    # # Save the figure
    # plt.savefig('../plots/Learning-Unlearning.png')


if __name__ == "__main__":
    opt = get_args()
    train(opt)
