import numpy as np

## FUNCTIONS TAKEN FROM https://www.kaggle.com/gmhost/gru-capsule
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')[:300]

def load_glove(opt, word_index):
    EMBEDDING_FILE = '../data/glove.840B.300d.txt'
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8"))

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = -0.005838499, 0.48782197
    embed_size = all_embs.shape[1]

    nb_words = min(opt.max_features, len(word_index) + 1)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= opt.max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_vector = embeddings_index.get(word.capitalize())
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix