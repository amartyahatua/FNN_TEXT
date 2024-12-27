import time
import torch
from tqdm.auto import tqdm
tqdm.pandas(desc='Progress')
# cross validation and metrics
import argparse
from learn_unlearn import get_learn_unlearn
from prepare_data import get_data
from train_model import train_model
from art.estimators.classification.pytorch import PyTorchClassifier
import torch.nn as nn
import numpy as np
from bilstm import BiLSTM


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document 
        Classification""")
    # New param
    parser.add_argument("--embed_size", type=int, default=300)
    parser.add_argument("--max_features", type=int, default=120000)
    parser.add_argument("--maxlen", type=int, default=750)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--SEED", type=int, default=10)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--learning_epoch", type=int, default=1)
    parser.add_argument("--unlearning_epoch", type=int, default=1)
    parser.add_argument("--n_turn", type=int, default=1)
    parser.add_argument("--ranks", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_epoches", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)

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


def text_fnn(opt):
    embedding_matrix, le, X_train, y_train, X_cv, y_cv, X_retain, y_retain, X_forget, y_forget, test_y = get_data(opt)

    # X_train = torch.tensor(X_train, dtype=torch.long).cuda()
    # y_train = torch.tensor(y_train, dtype=torch.long).cuda()
    # X_cv = torch.tensor(X_cv, dtype=torch.long).cuda()
    # y_cv = torch.tensor(y_cv, dtype=torch.long).cuda()

    X_train = torch.tensor(X_train, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_cv = torch.tensor(X_cv, dtype=torch.long)
    y_cv = torch.tensor(y_cv, dtype=torch.long)

    # Create Torch datasets
    train = torch.utils.data.TensorDataset(X_train, y_train)
    valid = torch.utils.data.TensorDataset(X_cv, y_cv)

    # Create Data Loaders
    train_loader = torch.utils.data.DataLoader(train, batch_size=opt.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=opt.batch_size, shuffle=False)

    # Create Torch datasets
    retain = torch.utils.data.TensorDataset(X_retain, y_retain)
    forget = torch.utils.data.TensorDataset(X_forget, y_forget)

    # Create Data Loaders
    retain_loader = torch.utils.data.DataLoader(retain, batch_size=opt.batch_size, shuffle=True)
    forget_loader = torch.utils.data.DataLoader(forget, batch_size=opt.batch_size, shuffle=False)

    # Initial training
    train_start_time = time.time()
    model = train_model(opt, le, embedding_matrix, train_loader, valid_loader, X_cv, test_y)
    training_time = time.time() - train_start_time
    print('Total training time={:.2f}s'.format(training_time))

    # Learning and Unlearning
    model = get_learn_unlearn(le, model, opt, X_cv, test_y, retain_loader, forget_loader, valid_loader)

    # model = BiLSTM(le, opt, embedding_matrix)
    # model.load_state_dict(torch.load('../model/imdb/bilstm_model_imdb.pth', map_location=torch.device("cpu")))

    model.eval()

    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    mlp_art_model = PyTorchClassifier(model=model, loss=criterion, optimizer=optimizer, input_shape=(opt.maxlen,),
                                      nb_classes=2)

    train_pred = np.array([np.argmax(arr) for arr in mlp_art_model.predict(X_train.astype(np.float32))])
    print('Base model Train accuracy: ', np.sum(train_pred == y_train) / len(y_train))

    test_pred = np.array([np.argmax(arr) for arr in mlp_art_model.predict(X_cv.astype(np.float32))])
    print('Base model Test accuracy: ', np.sum(test_pred == y_cv) / len(y_cv))


if __name__ == "__main__":
    opt = get_args()
    text_fnn(opt)
