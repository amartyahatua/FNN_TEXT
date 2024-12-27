import re
import torch
import pandas as pd
from embedding import load_glove
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

contraction_dict = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                    "could've": "could have", "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
                    "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                    "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did",
                    "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would",
                    "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am",
                    "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
                    "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                    "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is",
                    "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
                    "mightn't": "might not", "mightn't've": "might not have", "must've": "must have",
                    "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
                    "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not",
                    "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
                    "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
                    "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have",
                    "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have", "so's": "so as",
                    "this's": "this is", "that'd": "that would", "that'd've": "that would have", "that's": "that is",
                    "there'd": "there would", "there'd've": "there would have", "there's": "there is",
                    "here's": "here is", "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                    "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                    "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will",
                    "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not",
                    "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is",
                    "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did",
                    "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have",
                    "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",
                    "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have",
                    "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                    "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are",
                    "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have",
                    "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}


# Usage
def clean_text(x):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', x)
    return x


def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    return x


def get_data(opt):
    # data = get_data_drugsCom()
    data = get_data_IMDB()

    def _get_contractions(contraction_dict):
        contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
        return contraction_dict, contraction_re

    def replace_contractions(text):
        def replace(match):
            return contractions[match.group(0)]

        return contractions_re.sub(replace, text)

    contractions, contractions_re = _get_contractions(contraction_dict)

    # lower the text
    data["text"] = data["text"].apply(lambda x: x.lower())
    # Clean the text
    data["text"] = data["text"].apply(lambda x: clean_text(x))
    # Clean numbers
    data["text"] = data["text"].apply(lambda x: clean_numbers(x))
    # Clean Contractions
    data["text"] = data["text"].apply(lambda x: replace_contractions(x))

    train_X, test_X, train_y, test_y = train_test_split(data['text'], data['class'],
                                                        stratify=data['class'],
                                                        test_size=0.25)

    train_X, test_X, train_y, test_y = train_X.iloc[0:100], test_X.iloc[0:100], train_y.iloc[0:100], test_y.iloc[0:100]
    print("Train shape : ", train_X.shape)
    print("Test shape : ", test_X.shape)

    # Tokenize the sentences
    tokenizer = Tokenizer(num_words=opt.max_features)
    tokenizer.fit_on_texts(list(train_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    test_X = tokenizer.texts_to_sequences(test_X)

    # Pad the sentences
    train_X = pad_sequences(train_X, maxlen=opt.maxlen)
    test_X = pad_sequences(test_X, maxlen=opt.maxlen)

    le = LabelEncoder()
    train_y = le.fit_transform(train_y.values)
    test_y = le.transform(test_y.values)

    embedding_matrix = load_glove(opt, tokenizer.word_index)

    # X_train = torch.tensor(train_X, dtype=torch.long).cuda()
    # y_train = torch.tensor(train_y, dtype=torch.long).cuda()
    # X_cv = torch.tensor(test_X, dtype=torch.long).cuda()
    # y_cv = torch.tensor(test_y, dtype=torch.long).cuda()
    X_train = torch.tensor(train_X, dtype=torch.long)
    y_train = torch.tensor(train_y, dtype=torch.long)
    X_cv = torch.tensor(test_X, dtype=torch.long)
    y_cv = torch.tensor(test_y, dtype=torch.long)

    # Create Torch datasets
    # train = torch.utils.data.TensorDataset(x_train, y_train)
    # valid = torch.utils.data.TensorDataset(x_cv, y_cv)
    #
    # # Create Data Loaders
    # train_loader = torch.utils.data.DataLoader(train, batch_size=opt.batch_size, shuffle=True)
    # valid_loader = torch.utils.data.DataLoader(valid, batch_size=opt.batch_size, shuffle=False)

    # Divide the dataset into forget and retain set
    X_retain, X_forget, y_retain, y_forget = train_test_split(X_train, y_train, random_state=104, test_size=0.25,
                                                              shuffle=True)
    # Create Torch datasets
    # retain = torch.utils.data.TensorDataset(X_retain, y_retain)
    # forget = torch.utils.data.TensorDataset(X_forget, y_forget)
    #
    # # Create Data Loaders
    # retain_loader = torch.utils.data.DataLoader(retain, batch_size=opt.batch_size, shuffle=True)
    # forget_loader = torch.utils.data.DataLoader(forget, batch_size=opt.batch_size, shuffle=False)

    return embedding_matrix, le, X_train, y_train, X_cv, y_cv, X_retain, y_retain, X_forget, y_forget, test_y

def get_data_drugsCom():
    data1 = pd.read_csv("../data/drugsComTrain_raw.csv")
    data2 = pd.read_csv("../data/drugsComTest_raw.csv")
    data = pd.concat([data1, data2])[['review', 'condition']]

    # remove NULL Values from data
    data = data[pd.notnull(data['review'])]
    data = data[data['condition'] != 'OTHER']
    data = data.rename(columns={'review': 'text', 'condition': 'class'})
    return data

def get_data_IMDB():
    data = pd.read_csv("../data/IMDB Dataset.csv")

    # remove NULL Values from data
    data = data[pd.notnull(data['review'])]
    data = data.rename(columns={'review': 'text', 'sentiment': 'class'})
    return data