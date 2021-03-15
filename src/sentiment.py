import pickle

import numpy as np
import tensorflow as tf
from deeppavlov.models.embedders.elmo_embedder import ELMoEmbedder
from nltk.tokenize import word_tokenize
from tensorflow import keras
from tensorflow.keras import backend as K

from .config import MAX_LENGTH, EMB_SIZE, BATCH_SiZE, CLASS_DICT

from natasha import (
    Segmenter,
    MorphVocab,
    
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    
    PER,
    NamesExtractor,

    Doc
    )


class SentSegmenter:
        
    def __init__(self, text):
        self.doc = Doc(text)
        #self.text = [token.text for token in self.doc.tokens]
        self.tokens_list = []
        self.tokenize()
        #print('    tokenized')
        self.divide()
        #print('    divided')

    def divide(self):
        # list of jsons
        self.sents = []
        self.sents2jsons()

    def tokenize(self):
        self.doc.segment(segmenter)
        self.doc.tag_morph(morph_tagger)
        self.doc.tag_ner(ner_tagger)
        self.doc.parse_syntax(syntax_parser)
        self.doc2sents()
        self.texts = [token.text for tokens in self.tokens_list for token in tokens]

    # returns the only tokens
    # that have needed rel
    # and are not used before
    def rel_word_tokens(self, sent, rel=('root'), used_ids=[]):
        res = []
        for word in sent:
            #print(  'word =', word)
            if word.rel.split(':')[0] in rel and word.id not in used_ids:
                res.append(word)
        return res

    # list of lists to flat list
    def flat_list(self, ll):
        return [item for list in ll for item in list]

    # returns a text list by the tokens
    def tokens2txt(self, tl):
        wl = [w.text for w in tl]
        res = ' '.join(wl)
        return res

    # this is not exactly what we're needed for
    # it should take in a sent with the subject
    # remember: subj --commits--> obj
    def sentence_division(self, sent, rel=('root'), used_ids=[], return_used_ids=False):

        rel_words = self.rel_word_tokens(sent, rel, used_ids)
        rel_ids = [w.id for w in rel_words]
        #print('    rel words:', [w.text for w in rel_words])
        root_words = self.rel_word_tokens(sent, 'root')
        #print('    root words:', [w.text for w in root_words])
        #print(root_ids)
        if rel_words == [] and root_words == []:
            if return_used_ids:
                return [[w for w in sent if w.id not in used_ids]], []
            else:
                return [[w for w in sent if w.id not in used_ids]]

        rel_ids = list(set(rel_ids) - set(used_ids))
        sent_parts = [[w] for w in rel_words]
        sent_parts_ids = [[w.id] for w in rel_words]
        root_ids = [w.id for w in root_words]
        if rel == ('root'):
            root_ids = [w.head_id for w in root_words]
        #root_ids.extend(used_ids)

        #print('  rel ids =', rel_ids)
        #print('  root ids =', root_ids)
        #print('  used ids =', used_ids)
        
        for s in sent:
            branch = []
            branch_ids = []
            if s.id in rel_ids or s.id in used_ids:
                #print('  skip:', s.id)
                continue
            si = s
            #print('si.id =', si.id, si.id not in root_ids)
            while si.id not in rel_ids and si.id not in root_ids and si.id not in used_ids:#(not (True in [si.id in spii for spii in sent_parts_ids])):# and (si.head_id not in root_ids):
                #print(' si.id = ', si.id, ' ', si.id not in root_ids)
                branch.append(si)
                branch_ids.append(si.id)
                # searching new si
                for ns in sent:
                #print('si.head_id =', si.head_id)
                    if si.head_id == ns.id:
                        si = ns
                        break
            
            if si.id in root_ids or si.id in used_ids:
                #print('  skip:', si.id)
                root_ids.extend(branch_ids)
                continue
            #print('    branch =', [w.text for w in branch])
            # adding the branch to its sent_part
            for i in range(len(sent_parts)):
                #print(sent_parts_ids[i])
                if si.id in sent_parts_ids[i]:
                    sent_parts[i].extend(branch)
                    sent_parts_ids[i].extend(branch_ids)

        if return_used_ids == True:
            return [[word for word in sent if word in part] for part in sent_parts], list(set(self.flat_list(sent_parts_ids)))
        return [[word for word in sent if word in part] for part in sent_parts]

    # returns json generator
    # by the token list
    def sents2jsons(self):#, sents):

        for token_sent in self.tokens_list:

            root_parts = self.sentence_division(token_sent)
            word_bags = [[token.text for token in s] for s in root_parts]
            root_texts = [' '.join(part) for part in word_bags]
            #print('    root parts:', root_parts)

            for root_part_text in root_parts:
                # obj
                param = ('obj')
                obj_parts, obj_ids = self.sentence_division(root_part_text, param, return_used_ids=True)
                obj_word_bags = [[token.text for token in s if token.rel != 'punct'] for s in obj_parts]
                obj_word_bag = [word for sent in obj_word_bags for word in sent]
                obj_texts = [' '.join(part) for part in obj_word_bags]

                # nsubj
                param = ('nsubj')
                nsubj_parts, nsubj_ids = self.sentence_division(root_part_text, param, used_ids=obj_ids, return_used_ids=True)
                nsubj_word_bags = [[token.text for token in s if token.rel != 'punct'] for s in nsubj_parts]
                nsubj_word_bag = [word for sent in nsubj_word_bags for word in sent]
                nsubj_texts = [' '.join(part) for part in nsubj_word_bags]

                # commit
                commit_parts = self.sentence_division(root_part_text, used_ids=obj_ids + nsubj_ids)
                commit_word_bags = [[token.text for token in s if token.rel != 'punct'] for s in commit_parts]
                commit_word_bag = [word for sent in commit_word_bags for word in sent]
                #print('commit word bag 0:', [word for sent in word_bags for word in sent])
                commit_texts = [' '.join(part) for part in commit_word_bags]

                ans = {'obj' : obj_word_bag,
                        'nsubj' : nsubj_word_bag,
                        'commit' : commit_word_bag}
                #print('    ans:', ans)
                
                #yield ans
                self.sents.append(ans)

    def doc2sents(self):
        self.tokens_list = [[word for word in self.doc.sents[n].tokens] for n in range(len(self.doc.sents))]

    def print_sents(self):
        for sent in self.sents:
            print(sent)

    def return_sents(self):
        return self.sents

    def print_texts(self):
        print(self.texts)
    
    def return_spec_rel(self, rels=('nsubj', 'commit')):
        result = []
        for sent in self.sents:
            #print('    sent:', sent)
            sent_res = []
            for rel in rels:
                sent_res.extend(sent[rel])
            txt = [word for word in self.texts if word in sent_res]
            result.extend(txt)
            #print('    txt:', txt)
        return result


def load_models():
    global SESS

    global ELMO_COMMENTS
    global ELMO_NEWS

    global COMMENTS_CNN_MODEL_GRAPH
    global COMMENTS_CNN_MODEL
    global NEWS_CNN_MODEL

    global TEXT_TYPE_MODEL

    # Natasha
    global segmenter
    global morph_vocab

    global morph_tagger
    global syntax_parser
    global ner_tagger

    global names_extractor

    segmenter = Segmenter()
    morph_vocab = MorphVocab()

    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)
    syntax_parser = NewsSyntaxParser(emb)
    ner_tagger = NewsNERTagger(emb)

    names_extractor = NamesExtractor(morph_vocab)
    # end Natasha

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
    # original segmenter
    '''
    clean_words = [w.strip(',-\":;') for w in word_tokenize(text)]
    clean_words = [w for w in clean_words if w]
    '''
    
    # new segmenter
    
    segmented_text = SentSegmenter(text)

    rels = ('nsubj', 'commit')
    clean_words = segmented_text.return_spec_rel(rels)
    
    #print('clean words:', clean_words)

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
