import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from art.utils import load_nursery
from sklearn.ensemble import RandomForestClassifier
from art.estimators.classification.scikitlearn import ScikitlearnRandomForestClassifier
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from art.estimators.classification.pytorch import PyTorchClassifier
import numpy as np

(x_train, y_train), (x_test, y_test), _, _ = load_nursery(test_set=0.5)
model = RandomForestClassifier()
model.fit(x_train, y_train)

art_classifier = ScikitlearnRandomForestClassifier(model)

print('Base model accuracy: ', model.score(x_test, y_test))

from art.attacks.inference.membership_inference import MembershipInferenceBlackBox

attack_train_ratio = 0.5
attack_train_size = int(len(x_train) * attack_train_ratio)
attack_test_size = int(len(x_test) * attack_train_ratio)

bb_attack = MembershipInferenceBlackBox(art_classifier)

# train attack model
bb_attack.fit(x_train[:attack_train_size], y_train[:attack_train_size],
              x_test[:attack_test_size], y_test[:attack_test_size])

# reduce size of training set to make attack slightly better
train_set_size = 500
x_train = x_train[:train_set_size]
y_train = y_train[:train_set_size]
x_test = x_test[:train_set_size]
y_test = y_test[:train_set_size]
attack_train_size = int(len(x_train) * attack_train_ratio)
attack_test_size = int(len(x_test) * attack_train_ratio)


class ModelToAttack(nn.Module):

    def __init__(self, num_classes, num_features):
        super(ModelToAttack, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.Tanh(), )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Tanh(), )

        self.fc3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(), )

        self.fc4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return self.classifier(out)


mlp_model = ModelToAttack(4, 24)
mlp_model = torch.nn.DataParallel(mlp_model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.01)


def calc_precision_recall(predicted, actual, positive_value=1):
    score = 0  # both predicted and actual are positive
    num_positive_predicted = 0  # predicted positive
    num_positive_actual = 0  # actual positive
    for i in range(len(predicted)):
        if predicted[i] == positive_value:
            num_positive_predicted += 1
        if actual[i] == positive_value:
            num_positive_actual += 1
        if predicted[i] == actual[i]:
            if predicted[i] == positive_value:
                score += 1

    if num_positive_predicted == 0:
        precision = 1
    else:
        precision = score / num_positive_predicted  # the fraction of predicted “Yes” responses that are correct
    if num_positive_actual == 0:
        recall = 1
    else:
        recall = score / num_positive_actual  # the fraction of “Yes” responses that are predicted correctly

    return precision, recall

class NurseryDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = torch.from_numpy(x.astype(np.float64)).type(torch.FloatTensor)

        if y is not None:
            self.y = torch.from_numpy(y.astype(np.int8)).type(torch.LongTensor)
        else:
            self.y = torch.zeros(x.shape[0])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if idx >= len(self.x):
            raise IndexError("Invalid Index")

        return self.x[idx], self.y[idx]


train_set = NurseryDataset(x_train, y_train)
train_loader = DataLoader(train_set, batch_size=100, shuffle=True, num_workers=0)

for epoch in range(20):
    for (input, targets) in train_loader:
        input, targets = torch.autograd.Variable(input), torch.autograd.Variable(targets)

        optimizer.zero_grad()
        outputs = mlp_model(input)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

mlp_art_model = PyTorchClassifier(model=mlp_model, loss=criterion, optimizer=optimizer, input_shape=(24,), nb_classes=4)

train_pred = np.array([np.argmax(arr) for arr in mlp_art_model.predict(x_train.astype(np.float32))])
print('Base model Train accuracy: ', np.sum(train_pred == y_train) / len(y_train))

test_pred = np.array([np.argmax(arr) for arr in mlp_art_model.predict(x_test.astype(np.float32))])
print('Base model Test accuracy: ', np.sum(test_pred == y_test) / len(y_test))

mlp_attack_bb = MembershipInferenceBlackBox(mlp_art_model, attack_model_type='rf')

# train attack model
mlp_attack_bb.fit(x_train[:attack_train_size].astype(np.float32), y_train[:attack_train_size],
              x_test[:attack_test_size].astype(np.float32), y_test[:attack_test_size])

# infer
mlp_inferred_train_bb = mlp_attack_bb.infer(x_train[attack_train_size:].astype(np.float32), y_train[attack_train_size:])
mlp_inferred_test_bb = mlp_attack_bb.infer(x_test[attack_test_size:].astype(np.float32), y_test[attack_test_size:])

# check accuracy
mlp_train_acc_bb = np.sum(mlp_inferred_train_bb) / len(mlp_inferred_train_bb)
mlp_test_acc_bb = 1 - (np.sum(mlp_inferred_test_bb) / len(mlp_inferred_test_bb))
mlp_acc_bb = (mlp_train_acc_bb * len(mlp_inferred_train_bb) + mlp_test_acc_bb * len(mlp_inferred_test_bb)) / (len(mlp_inferred_train_bb) + len(mlp_inferred_test_bb))

print(f"Members Accuracy: {mlp_train_acc_bb:.4f}")
print(f"Non Members Accuracy {mlp_test_acc_bb:.4f}")
print(f"Attack Accuracy {mlp_acc_bb:.4f}")

print(calc_precision_recall(np.concatenate((mlp_inferred_train_bb, mlp_inferred_test_bb)),
                            np.concatenate((np.ones(len(mlp_inferred_train_bb)), np.zeros(len(mlp_inferred_test_bb))))))