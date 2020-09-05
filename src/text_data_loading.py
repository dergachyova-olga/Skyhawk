import logging
import json
import pandas
import numpy
import re
from sklearn import model_selection, preprocessing

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')


class TextDataLoader(object):

    def __init__(self, cfg, label_encoder=None):

        # store config
        self.cfg = cfg

        # initialize variables
        self.df = pandas.DataFrame()
        self.label_encoder = preprocessing.LabelEncoder() if label_encoder is None else label_encoder

    # -------------------- Classification methods  ---------------------- #

    def load_classification_data(self, origin):

        # choose btw train, test and inference data
        if origin == 'train':

            # load raw json files
            fake, _ = self.load_clean_preprocess(self.cfg.data.fake_train_data, 'fake')
            real, _ = self.load_clean_preprocess(self.cfg.data.real_train_data, 'real')

            # split data to train and valid, and prepare x (tests) and y (labels)
            return self.create_train_valid_dataset(fake, real)

        elif origin == 'test':

            # load raw json files
            fake, _ = self.load_clean_preprocess(self.cfg.data.fake_test_data, 'fake')
            real, _ = self.load_clean_preprocess(self.cfg.data.real_test_data, 'real')

            # prepare x (tests) and y (labels)
            return self.create_test_dataset(fake, real)

        elif origin == 'inference':

            # load raw json file
            return self.load_clean_preprocess(self.cfg.data.inference_data, 'inference')

    def load_clean_preprocess(self, file_path, label):

        # load json file content
        with open(file_path) as f:
            articles = json.load(f)

        # eliminate entries with errors (title is usually missing)
        invalid = [i for i, article in enumerate(articles) if not article['title']]
        logging.info("\t The following %s entries are invalid and will be eliminated: %s." %
                     (label, ', '.join(str(i) for i in invalid)))

        # extract valid entries as texts
        texts = [article['title'] + '.\n' + article['text'] for article in articles if article['title']]
        logging.info("\t %d %s entries were loaded" % (len(texts), label))

        # clean and pre-process texts
        texts = self.clean(texts)
        texts = self.preprocess(texts)

        return texts, invalid

    def create_train_valid_dataset(self, fake, real):

        # create texts and labels
        texts = fake + real
        labels = ['fake'] * len(fake) + ['real'] * len(real)

        # save texts and labels in a dataframe
        self.df['text'] = texts
        self.df['label'] = labels

        # split into training and validation data sets
        train_x, valid_x, train_y, valid_y = model_selection.train_test_split(self.df['text'], self.df['label'],
                                                                              test_size=self.cfg.data.valid_size)
        # encode labels
        self.label_encoder = self.label_encoder.fit(self.df['label'])
        train_y = self.label_encoder.transform(train_y)
        valid_y = self.label_encoder.transform(valid_y)

        return train_x, valid_x, train_y, valid_y

    def create_test_dataset(self, fake, real):

        # create texts
        test_x = fake + real

        # encode labels
        labels = ['fake'] * len(fake) + ['real'] * len(real)
        test_y = self.label_encoder.transform(labels)

        return test_x, test_y

    # -------------------- Clustering methods  ---------------------- #

    def load_clustering_data(self):

        # load raw json file content
        with open(self.cfg.data.cluster_data) as f:
            articles = json.load(f)

        # extract texts and labels
        texts = [article['title'] + '.\n' + article['text'] for article in articles]
        labels = [article['cluster_name'] for article in articles]

        # clean and pre-process texts
        texts = self.clean(texts)
        texts = self.preprocess(texts)
        cluster_x = texts

        # save texts and labels in a dataframe
        self.df['text'] = texts
        self.df['label'] = labels

        # encode labels
        cluster_y = self.label_encoder.fit_transform(labels)

        return cluster_x, cluster_y

    def group_by_label(self, labels):

        # group all entries (articles) by labels
        groups = dict()
        for label in numpy.unique(labels):
            groups[label] = numpy.where(labels == label)[0].tolist()

        return groups

    # -------------------- Common methods  ---------------------- #

    def clean(self, texts):

        # remove special characters (also removes punctuation and digits)
        texts = [re.sub('[^a-z]+', ' ', text) for text in texts]

        # remove any leftover single letters
        texts = [re.sub('\b[a-z]\b', ' ', text) for text in texts]

        # remove extra spaces
        texts = [re.sub(' +', ' ', text) for text in texts]

        return texts

    def preprocess(self, texts):

        # normalize case
        texts = [text.lower() for text in texts]

        # remove stop words
        if self.cfg.preprocessing.remove_stop_words:
            stop_words = stopwords.words('english')
            texts = [' '.join(word for word in text.split() if word not in stop_words) for text in texts]

        # lemmatize
        if self.cfg.preprocessing.lemmatize:
            lemmatizer = WordNetLemmatizer()
            texts = [' '.join(lemmatizer.lemmatize(word) for word in text.split()) for text in texts]

        return texts

    def decode_labels(self, y_):

        # get true label names instead of ids
        return self.label_encoder.inverse_transform(y_)