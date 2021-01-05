import pickle

import numpy as np
import tensorflow as tf
from deeppavlov.models.embedders.elmo_embedder import ELMoEmbedder
from nltk.tokenize import word_tokenize
from tensorflow import keras
from tensorflow.keras import backend as K

from .config import MAX_LENGTH, EMB_SIZE, BATCH_SiZE, CLASS_DICT


def load_models():
    global SESS

    global ELMO_COMMENTS
    global ELMO_NEWS

    global COMMENTS_CNN_MODEL_GRAPH
    global COMMENTS_CNN_MODEL
    global NEWS_CNN_MODEL

    global TEXT_TYPE_MODEL

    ELMO_COMMENTS = ELMoEmbedder("./model_comments/", elmo_output_names=['elmo'])
    ELMO_NEWS = ELMoEmbedder("./model_posts/", elmo_output_names=['elmo'])

    SESS = tf.Session()
    SESS.run(tf.global_variables_initializer())

    K.set_session(SESS)
    COMMENTS_CNN_MODEL = keras.models.load_model('comments_cnn_model.h5')
    NEWS_CNN_MODEL = keras.models.load_model("news_cnn_model.h5")

    with open("text_type_model.pkl", 'rb') as file:
        TEXT_TYPE_MODEL = pickle.load(file)


def convert_text(text):
    clean_words = [w.strip(',-\":;') for w in word_tokenize(text)]
    clean_words = [w for w in clean_words if w]
    return clean_words[:MAX_LENGTH]


def preprocessing(texts, elmo):
    X = list(map(convert_text, texts))
    embs = elmo(X)
    return embs


def get_pred(text_batch, elmo, cnn_model):
    embs = preprocessing(text_batch, elmo)
    matrix = np.zeros((len(embs), MAX_LENGTH, EMB_SIZE))
    for idx in range(len(embs)):
        matrix[idx, :len(embs[idx])] = embs[idx]
    pred = cnn_model.predict(matrix)
    return pred


def get_sentiment(texts, bs):
    for i in range(int(np.ceil(len(texts) / bs))):
        text_batch = texts[i * bs:(i + 1) * bs]
        answers = np.zeros((len(text_batch), len(CLASS_DICT)))
        types = np.array(TEXT_TYPE_MODEL.predict(text_batch))
        comments = np.array(text_batch)[types == 0]
        news = np.array(text_batch)[types == 1]

        with SESS.graph.as_default():
            K.set_session(SESS)
            if len(comments):
                pred_comments = get_pred(comments, ELMO_COMMENTS, COMMENTS_CNN_MODEL)
                answers[types == 0] = pred_comments
            if len(news):
                pred_news = get_pred(news, ELMO_NEWS, NEWS_CNN_MODEL)
                answers[types == 1] = pred_news

        yield [{CLASS_DICT[class_id]: score for class_id, score in enumerate(row)} for row in answers]


def evaluate_sentiment(texts):
    results = []
    for res in get_sentiment(texts, BATCH_SiZE):
        results.extend(res)
    return results


load_models()
