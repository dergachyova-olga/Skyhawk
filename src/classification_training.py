import config
import sys
import os
import logging
import operator
from data_loading import DataLoader
from feature_extraction import FeatureExtractor
from model import Model


# load configuration
default_config_folder = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'configs')
default_config_file = os.path.join(default_config_folder, 'classification_training.ini')
cfg = config.load_config(sys.argv[1] if len(sys.argv) > 1 else default_config_file)


logging.info('---------------------------------------------------')
logging.info('---------- Classification module (train) ----------')
logging.info('---------------------------------------------------')


# ---------------------------- Load data ---------------------------- #
data_loader = DataLoader(cfg)

logging.info('Loading train data...')
train_x, valid_x, train_y, valid_y = data_loader.load_classification_data('train')

logging.info('Loading test data...')
test_x, test_y = data_loader.load_classification_data('test')


# ----------------------- Extract features -------------------------- #
feature_extractor = FeatureExtractor(cfg, data_loader.df)

# Bag-of-Words
if cfg.features.bow:
    logging.info('Extracting BOW features...')

    # compute transformer to Bag-of-Words space
    trans_bow = feature_extractor.get_bow_transformer()

    # compute features
    train_bow = feature_extractor.get_bow_features(train_x, trans_bow)
    valid_bow = feature_extractor.get_bow_features(valid_x, trans_bow)
    test_bow = feature_extractor.get_bow_features(test_x, trans_bow)


# TF-IDF
if cfg.features.tfidf:
    logging.info('Extracting TF-IDF features...')

    # compute transformers to TF-IDF space
    trans_tfidf_word = feature_extractor.get_tfidf_transformer('word')
    trans_tfidf_ngram = feature_extractor.get_tfidf_transformer('ngram')
    trans_tfidf_char = feature_extractor.get_tfidf_transformer('char')

    # compute features
    train_tfidf_word = feature_extractor.get_tfidf_features(train_x, trans_tfidf_word)
    train_tfidf_ngram = feature_extractor.get_tfidf_features(train_x, trans_tfidf_ngram)
    train_tfidf_char = feature_extractor.get_tfidf_features(train_x, trans_tfidf_char)
    valid_tfidf_word = feature_extractor.get_tfidf_features(valid_x, trans_tfidf_word)
    valid_tfidf_ngram = feature_extractor.get_tfidf_features(valid_x, trans_tfidf_ngram)
    valid_tfidf_char = feature_extractor.get_tfidf_features(valid_x, trans_tfidf_char)
    test_tfidf_word = feature_extractor.get_tfidf_features(test_x, trans_tfidf_word)
    test_tfidf_ngram = feature_extractor.get_tfidf_features(test_x, trans_tfidf_ngram)
    test_tfidf_char = feature_extractor.get_tfidf_features(test_x, trans_tfidf_char)


# word embeddings
if cfg.features.word_embeddings:
    logging.info('Computing word embeddings...')

    # compute transformers to embedding space
    trans_emb = feature_extractor.get_embedding_transformer()
    train_emb = feature_extractor.get_embedding_features(train_x, trans_emb)
    valid_emb = feature_extractor.get_embedding_features(valid_x, trans_emb)
    test_emb = feature_extractor.get_embedding_features(test_x, trans_emb)


# -------------------- Train classifiers ---------------------------- #
classification_model = Model(cfg, data_loader.df)
models = dict()

if cfg.model.train_nb:
    logging.info('Training Naive Bayes classifier...')

    # train Naive Bayes on Bag-of-Words features
    if cfg.features.bow:
        models["NB_BOW"] = [classification_model.train('NB', train_bow, train_y), 'bow']

    # train Naive Bayes on TF-IDF features of different levels (word, ngram, and char)
    if cfg.features.tfidf:
        models["NB_TF-IDF_WORD"] = [classification_model.train('NB', train_tfidf_word, train_y), 'tfidf_word']
        models["NB_TF-IDF_NGRAM"] = [classification_model.train('NB', train_tfidf_ngram, train_y), 'tfidf_ngram']
        models["NB_TF-IDF_CHAR"] = [classification_model.train('NB', train_tfidf_char, train_y), 'tfidf_char']

    # train Naive Bayes on word embeddings
    if cfg.features.word_embeddings:
        models["NB_EMBEDDINGS"] = [classification_model.train('NB', train_emb, train_y), 'emb']


if cfg.model.train_lstm:
    logging.info('Training Long-Short Term Memory classifier...')

    # train LSTM on word embeddings
    if cfg.features.word_embeddings:
        models["LSTM_EMB"] = [classification_model.train('LSTM', train_emb, train_y,
                                                         feature_extractor.embeddings['embedding_matrix']), "emb"]


# ---------------------------- Predict ------------------------------ #
logging.info('Making predictions on validation data...')
predictions = dict()

# make predictions for every classifier-feature pair
for name, model in models.items():
    predictions[name] = classification_model.predict_class(model[0], eval('valid_' + model[1]))


# -------------- Evaluate models performance (valid data) ----------- #
logging.info('Evaluating models...')
scores = dict()

# compute performance scores for every classifier-feature pair
logging.info('Scores:')
for name, valid_y_ in predictions.items():
    scores[name] = classification_model.evaluate_classification(valid_y, valid_y_)
    logging.info('\t %s: accuracy = %.4f, precision = %.4f, recall = %.4f' %
                 (name, scores[name][0], scores[name][1], scores[name][2]))

# find best (by accuracy) classifier model
accuracy_scores = {k: v[0] for k, v in scores.items()}
best_model_name = max(accuracy_scores.items(), key=operator.itemgetter(1))[0]
best_model = models[best_model_name]
logging.info('Best model by accuracy is %s' % best_model_name)


# ------------------ Test best model (test data)  -------------------- #
logging.info('Testing best model...')
test_y_ = classification_model.predict_class(best_model[0], eval('test_' + best_model[1]))
score = classification_model.evaluate_classification(test_y, test_y_)
logging.info('\t Final scores: accuracy = %.4f, precision = %.4f, recall = %.4f' % (score[0], score[1], score[2]))


# ------------------------ Save best model --------------------------- #
logging.info('Saving best model...')
classification_model.save(best_model_name, best_model[0], best_model[1], eval('trans_' + best_model[1]),
                          feature_extractor.embeddings, data_loader.label_encoder)


logging.info('---------------------------------------------------')
logging.info('------------- Done! Have a nice day ;) ------------')
logging.info('---------------------------------------------------')