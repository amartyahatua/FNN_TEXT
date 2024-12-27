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
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist





# Step 1: Load the MNIST dataset
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
# x_train = np.expand_dims(x_train, axis=1)
# x_test = np.expand_dims(x_test, axis=1)
#
#
# # Step 1a: Swap axes to PyTorch's NCHW format
#
x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

# x_train = np.transpose(x_train, (0, 2, 1)).astype(np.float32)
# x_test = np.transpose(x_test, (0, 2, 1)).astype(np.float32)

# Step 2: Create the model

model = Net()

# Step 2a: Define the loss function and the optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# Step 3: Create the ART classifier


def train(model, x_train, y_train, epoch, type, turn, unlearning_type, device):
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(min_pixel_value, max_pixel_value),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10
    )

    model.train()
    model.type = type
    model.epoch = epoch
    model.turn = turn
    model.device = device
    model.unlearning_type=unlearning_type
    classifier.fit(x_train, y_train, batch_size=64, nb_epochs=epoch)
    return classifier


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
accuracy_list = []
mia_score_list = []

# 'rank_fr', 'ordered_fr', 'topK_fr', 'random_fr'
for unlearning_type in ['random_fr']:
    print(f'---------------------------------Unlearning type: {unlearning_type}---------------------------------')
    for turn in range(5):
        print(f'------------------------Turn = {turn}-----------------------------')
        print(f'------------------------Learning-----------------------------')
        # Step 4: Train the ART classifier
        for epoch in range(1, 4):

            classifier = train(model, x_train, y_train, epoch, 'learning', turn, '', device)
            # Step 5: Evaluate the ART classifier on benign test examples
            predictions = classifier.predict(x_test)
            accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
            print("Accuracy on benign test examples: {:0.2f}%".format(accuracy * 100))
            accuracy_list.append(accuracy)

            # Step 6: Generate adversarial test examples
            attack = FastGradientMethod(estimator=classifier, eps=0.2)
            x_test_adv = attack.generate(x=x_test)

            # Step 7: Evaluate the ART classifier on adversarial test examples
            predictions = classifier.predict(x_test_adv)
            accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
            mia_score_list.append(accuracy)
            print("Accuracy on adversarial test examples: {:0.2f}%".format(accuracy * 100))

        print(f'------------------------Unlearning-----------------------------')
        for epoch in range(1, 7):
            X_retain, X_forget, y_retain, y_forget = train_test_split(x_train, y_train, random_state=104, test_size=0.25,
                                                                      shuffle=True)
            classifier = train(model, x_train, y_train, epoch, 'unlearning', turn, unlearning_type, device)
            # Step 5: Evaluate the ART classifier on benign test examples
            predictions = classifier.predict(x_test)
            accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
            print("Accuracy on benign test examples: {:0.2f}%".format(accuracy * 100))
            accuracy_list.append(accuracy)

            # Step 6: Generate adversarial test examples
            attack = FastGradientMethod(estimator=classifier, eps=0.2)
            x_test_adv = attack.generate(x=X_forget)

            # Step 7: Evaluate the ART classifier on adversarial test examples
            predictions = classifier.predict(x_test_adv)
            accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_forget, axis=1)) / len(y_forget)
            mia_score_list.append(accuracy)
            print("Accuracy on adversarial test examples: {:0.2f}%".format(accuracy * 100))

