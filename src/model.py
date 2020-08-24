import os
import pickle
import numpy
from sklearn import naive_bayes
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score, v_measure_score
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.cluster import KMeans
from keras import layers, models, optimizers


class Model(object):

    def __init__(self, cfg, df=None):

        # store config
        self.cfg = cfg

        # store data
        self.df = df

    # ------------------ Classification methods  -------------------- #

    def train(self, classifier, x, y, embedding_matrix=None):

        # Naive Bayes
        if classifier == 'NB':
            return naive_bayes.MultinomialNB().fit(x, y)

        # Long-Short Term Memory neural network
        elif classifier == 'LSTM':
            model = self.create_LSTM_network(embedding_matrix)
            model.fit(x, y, epochs=self.cfg.lstm.epochs, verbose=0)
            return model

    def predict_class(self, model, x):

        # predict classes
        y_ = model.predict(x)

        # choose a class if prediction is a probability
        if len(numpy.unique(numpy.array(y_))) > 2:
            y_ = numpy.round(y_).astype('int64').flatten()

        return y_

    def create_LSTM_network(self, embedding_matrix):

        # layers
        input_layer = layers.Input((None,))
        embedding_layer = layers.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                           weights=[embedding_matrix], trainable=False)(input_layer)
        lstm_layer = layers.LSTM(self.cfg.lstm.lstm_neurons, dropout=0.2)(embedding_layer)
        dense_layer = layers.Dense(self.cfg.lstm.dense_neurons, activation="relu")(lstm_layer)
        dropout_layer = layers.Dropout(0.2)(dense_layer)
        output_layer = layers.Dense(1, activation="sigmoid")(dropout_layer)

        # model
        model = models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

        return model

    def evaluate_classification(self, y, y_):

        # compute performance scores for classifiers based on their predictions
        accuracy = accuracy_score(y_, y)
        precision = precision_score(y_, y)
        recall = recall_score(y_, y)

        return accuracy, precision, recall

        # -------------------- Clustering methods  ---------------------- #

    def lda(self, x):

        # compute LDA model
        model = LDA(n_components=self.cfg.lda.topics)
        model.fit(x)

        # compute LDA topic distributions as features
        features = model.transform(x)

        return model, features

    def kmeans(self, x, y):

        # compute k-means clusters
        model = KMeans(n_clusters=len(set(y)))
        model.fit(x)

        return model

    def predict_cluster(self, model, model_name, x):

        # cluster according to the algorithm
        if model_name == 'lda':
            y_ = numpy.argmax(model.transform(x), axis=-1)

        elif model_name == 'kmeans':
            y_ = model.predict(x)

        return y_

    def evaluate_clustering(self, y, y_):

        # compute Adjusted Rand Index
        # (0.0 - random clustering, 1.0 - clusters are identical)
        ari = adjusted_rand_score(y, y_)

        # compute Adjusted Mutual Info
        # (0.0 - random clustering, 1.0 - clusters are identical)
        ami = adjusted_mutual_info_score(y, y_)

        # compute V-measure
        # (0.0 - random clustering, 1.0 - clusters are identical)
        v = v_measure_score(y, y_)

        return ari, ami, v

    # ---------------------- Common methods  ------------------------ #

    def load(self):

        # load an object containing all information about the model
        input_file = open(self.cfg.model.path, 'rb')
        load_object = pickle.load(input_file)
        input_file.close()

        # load a keras model if needed
        model = models.load_model(load_object['path']) if load_object['path'] else load_object['model']

        return load_object['name'], model, load_object['feature'], load_object['trasformer'], \
               load_object['embeddings'], load_object['label_encoder'],

    def save(self, name, model, feature, trasformer, embeddings, label_encoder):

        # create output folder if doesn't exist
        if not os.path.exists(self.cfg.output.model_folder):
            os.mkdir(self.cfg.output.model_folder)

        # create an object that contains all necessary information to be saved
        save_object = {'path': '', 'name': name, 'model': model, 'feature': feature, 'trasformer': trasformer,
                       'embeddings': embeddings, 'label_encoder': label_encoder}

        # save a keras model separately if needed
        if isinstance(save_object['model'], models.Model):
            # save keras object and store its path
            save_object['path'] = os.path.join(self.cfg.output.model_folder, self.cfg.output.model_name + '.keras')
            save_object['model'].save(save_object['path'])
            save_object['model'] = None

        # pickle the object
        output_file = open(os.path.join(self.cfg.output.model_folder, self.cfg.output.model_name + '.pickle'), 'wb')
        pickle.dump(save_object, output_file)
        output_file.close()
