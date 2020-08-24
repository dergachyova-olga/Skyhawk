from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import gensim.models.keyedvectors as word2vec
import keras.preprocessing
import numpy


class FeatureExtractor(object):

    def __init__(self, cfg, df=None, embeddings=None):

        # store config
        self.cfg = cfg

        # store data
        self.df = df

        # set variables
        if embeddings is None and hasattr(self.cfg, 'word_embeddings'):
            self.embeddings = {'embedding_matrix': None, 'max_text_length': self.cfg.word_embeddings.max_text_length}
        else:
            self.embeddings = embeddings

    # ---------------------- Bag-of-Words  ------------------------ #

    def get_bow_transformer(self):

        # compute Bag-of-Words transformer
        transformer = CountVectorizer(analyzer='word', max_features=self.cfg.bow.max_features)
        return transformer.fit(self.df['text'])

    def get_bow_features(self, x, transformer):
        return transformer.transform(x)

    # ------------------------- IF-IDF  --------------------------- #

    def get_tfidf_transformer(self, level):

        # initialize TF-IDF transformer for appropriate level
        if level == 'word':
            transformer = TfidfVectorizer(analyzer='word', max_features=self.cfg.tfidf.max_features)

        elif level == 'ngram':
            transformer = TfidfVectorizer(analyzer='word', ngram_range=(self.cfg.tfidf.ngram_n, self.cfg.tfidf.ngram_n),
                                          max_features=self.cfg.tfidf.max_features)
        elif level == 'char':
            transformer = TfidfVectorizer(analyzer='char', ngram_range=(self.cfg.tfidf.ngram_n, self.cfg.tfidf.ngram_n),
                                          max_features=self.cfg.tfidf.max_features)

        # compute TF-IDF transformer
        return transformer.fit(self.df['text'])

    def get_tfidf_features(self, x, transformer):
        return transformer.transform(x)

    # --------------------- Word embeddings  ---------------------- #

    def get_embedding_transformer(self):

        # create and fit a tokenizer
        transformer = keras.preprocessing.text.Tokenizer()
        transformer.fit_on_texts(self.df['text'])
        return transformer

    def get_embedding_features(self, x, transformer):

        # load pre-trained word embeddings if not done yet
        if self.embeddings['embedding_matrix'] is None:
            self.load_word_embeddings(transformer.word_index)

        # convert text to sequence of tokens and pad
        seq = transformer.texts_to_sequences(x)
        features = keras.preprocessing.sequence.pad_sequences(seq, maxlen=self.embeddings['max_text_length'])

        return features

    def load_word_embeddings(self, word_index):

        # load pre-trained word embeddings
        pretrained_embeddings = word2vec.KeyedVectors.load_word2vec_format(self.cfg.word_embeddings.path, binary=True)
        embedding_keys = pretrained_embeddings.vocab.keys()

        # map tokens to vectors and populate embedding matrix
        self.embeddings['embedding_matrix'] = numpy.zeros((len(word_index) + 1, pretrained_embeddings.vector_size))
        for word, i in word_index.items():
            if word in embedding_keys:
                self.embeddings['embedding_matrix'][i] = pretrained_embeddings[word]

    # ---------------- Latent Semantic Analysis ------------------ #

    def lsa(self, x):
        # reduce dimensions
        lsa = TruncatedSVD(n_components=self.cfg.lsa.components)
        return lsa.fit_transform(x)

    # --------------------- Common methods  ---------------------- #

    def get_features(self, feature, transformer, x):

        # get features according to the name
        if feature == 'bow':
            return self.get_bow_features(x, transformer)
        elif feature == 'tfidf_word' or feature == 'tfidf_ngram' or feature == 'tfidf_char':
            return self.get_bow_features(x, transformer)
        elif feature == 'emb':
            return self.get_embedding_features(x, transformer)


