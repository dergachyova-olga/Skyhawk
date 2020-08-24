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
default_config_file = os.path.join(default_config_folder, 'clustering.ini')
cfg = config.load_config(sys.argv[1] if len(sys.argv) > 1 else default_config_file)


logging.info('---------------------------------------------------')
logging.info('----------------- Clustering module ---------------')
logging.info('---------------------------------------------------')


# ---------------------------- Load data ---------------------------- #
data_loader = DataLoader(cfg)

logging.info('Loading clustering data...')
data_x, data_y = data_loader.load_clustering_data()


# ----------------------- Extract features -------------------------- #
feature_extractor = FeatureExtractor(cfg, data_loader.df)

# Bag-of-Words
if cfg.features.bow:
    logging.info('Extracting BOW features...')
    trans_bow = feature_extractor.get_bow_transformer()
    data_bow = feature_extractor.get_bow_features(data_x, trans_bow)


# TF-IDF
if cfg.features.tfidf:
    logging.info('Extracting TF-IDF features...')

    # compute transformers to TF-IDF space
    trans_tfidf_word = feature_extractor.get_tfidf_transformer('word')
    trans_tfidf_ngram = feature_extractor.get_tfidf_transformer('ngram')
    trans_tfidf_char = feature_extractor.get_tfidf_transformer('char')

    # compute features
    data_tfidf_word = feature_extractor.get_tfidf_features(data_x, trans_tfidf_word)
    data_tfidf_ngram = feature_extractor.get_tfidf_features(data_x, trans_tfidf_ngram)
    data_tfidf_char = feature_extractor.get_tfidf_features(data_x, trans_tfidf_char)


# word embeddings
if cfg.features.word_embeddings:
    logging.info('Computing word embeddings...')
    trans_emb = feature_extractor.get_embedding_transformer()
    data_emb = feature_extractor.get_embedding_features(data_x, trans_emb)


# ----------------------- Reduce dimensionality ----------------------- #
if cfg.features.lsa:
    logging.info('Reducing dimensionality...')

    # Bag-of-Words
    if cfg.features.bow:
        data_bow_lsa = feature_extractor.lsa(data_bow)

    # TF-IDF
    if cfg.features.tfidf:
        data_tfidf_word_lsa = feature_extractor.lsa(data_tfidf_word)
        data_tfidf_ngram_lsa = feature_extractor.lsa(data_tfidf_ngram)
        data_tfidf_char_lsa = feature_extractor.lsa(data_tfidf_char)


# ----------------------- Compute topics (LDA) ------------------------ #
clustering_model = Model(cfg, data_loader.df)

if cfg.model.lda:
    logging.info('Latent Dirichlet Allocation...')

    # compute LDA on Bag-of-Words features
    if cfg.features.bow:
        _, data_bow_lda = clustering_model.lda(data_bow)

    # compute LDA on TF-IDF features (words and n-grams only)
    if cfg.features.tfidf:
        _, data_tfidf_word_lda = clustering_model.lda(data_tfidf_word)
        _, data_tfidf_ngram_lda = clustering_model.lda(data_tfidf_ngram)


# ------------------------- Compute clusters ------------------------- #
models = dict()

# K-means
if cfg.model.kmeans:
    logging.info('Computing clusters using K-means algorithm...')

    # K-means clustering on Bag-of-Words features
    if cfg.features.bow:
        models["KMEANS_BOW"] = [clustering_model.kmeans(data_bow, data_y), 'kmeans', 'bow']

    # K-means clustering on TF-IDF features
    if cfg.features.tfidf:
        models["KMEANS_TF-IDF_WORD"] = [clustering_model.kmeans(data_tfidf_word, data_y), 'kmeans', 'tfidf_word']
        models["KMEANS_TF-IDF_NGRAM"] = [clustering_model.kmeans(data_tfidf_ngram, data_y), 'kmeans', 'tfidf_ngram']
        models["KMEANS_TF-IDF_CHAR"] = [clustering_model.kmeans(data_tfidf_char, data_y), 'kmeans', 'tfidf_char']

    # K-means clustering on Bag-of-Words features + LSA
    if cfg.features.bow and cfg.features.lsa:
        models["KMEANS_BOW_LSA"] = [clustering_model.kmeans(data_bow_lsa, data_y), 'kmeans', 'bow_lsa']

    # K-means clustering on TF-IDF features + LSA
    if cfg.features.tfidf and cfg.features.lsa:
        models["KMEANS_TF-IDF_WORD_LSA"] = [clustering_model.kmeans(data_tfidf_word_lsa, data_y), 'kmeans', 'tfidf_word_lsa']
        models["KMEANS_TF-IDF_NGRAM_LSA"] = [clustering_model.kmeans(data_tfidf_ngram_lsa, data_y), 'kmeans', 'tfidf_ngram_lsa']
        models["KMEANS_TF-IDF_CHAR_LSA"] = [clustering_model.kmeans(data_tfidf_char_lsa, data_y), 'kmeans', 'tfidf_char_lsa']

    # K-means clustering on Bag-of-Words features + LDA
    if cfg.features.bow and cfg.model.lda:
        models["KMEANS_BOW_LDA"] = [clustering_model.kmeans(data_bow_lda, data_y), 'kmeans', 'bow_lda']

    # K-means clustering on TF-IDF features + LDA
    if cfg.features.tfidf and cfg.model.lda:
        models["KMEANS_TF-IDF_WORD_LDA"] = [clustering_model.kmeans(data_tfidf_word_lda, data_y), 'kmeans', 'tfidf_word_lda']
        models["KMEANS_TF-IDF_NGRAM_LDA"] = [clustering_model.kmeans(data_tfidf_ngram_lda, data_y), 'kmeans', 'tfidf_ngram_lda']

    # K-means clustering on word embeddings
    if cfg.features.bow:
        models["KMEANS_EMB"] = [clustering_model.kmeans(data_emb, data_y), 'kmeans', 'emb']


# -------------------------- Cluster data ---------------------------- #
logging.info('Clustering data...')
predictions = dict()

# cluster data for every algorithm-feature pair
for name, model in models.items():
    predictions[name] = clustering_model.predict_cluster(model[0], model[1], eval('data_' + model[2]))


# ----------------------- Evaluate results ----------------------------- #
logging.info('Evaluating models...')
scores = dict()

# print predicted clusters for every algorithm-feature pair
for model_name, data_y_ in predictions.items():
    scores[model_name] = clustering_model.evaluate_clustering(data_y, data_y_)
    logging.info('\t %s model:  ARI = %.4f, AMI = %.4f, V = %.4f' %
                 (model_name, scores[model_name][0], scores[model_name][1], scores[model_name][2]))

# find best (by V-measure) clustering model
v_measure_scores = {k: v[-1] for k, v in scores.items()}
best_model_name = max(v_measure_scores.items(), key=operator.itemgetter(1))[0]
best_model = models[best_model_name]
logging.info('Best model is %s: v-measure = %.4f' % (best_model_name, v_measure_scores[best_model_name]))


# ---------------------- Visualize results ---------------------------- #
if cfg.output.visualize:
    logging.info('Printing results...')

    # print original clusters
    logging.info('--- Original clusters:')
    for cluster, entries in data_loader.group_by_label(data_y).items():
        cluster_name = str(data_loader.decode_labels([cluster])[0])
        logging.info('\t\t %s: %s' % (cluster_name, ', '.join(map(str, entries))))

    # print predicted clusters for every algorithm-feature pair
    logging.info('--- Predictions from %s model:' % best_model_name)
    for cluster, entries in data_loader.group_by_label(predictions[best_model_name]).items():
        logging.info('\t\t %s: %s' % (cluster, ', '.join(map(str, entries))))


# ------------------------ Save best model --------------------------- #
logging.info('Saving best model...')
clustering_model.save(best_model_name, best_model[0], best_model[2], None, None, data_loader.label_encoder)


logging.info('---------------------------------------------------')
logging.info('------------- Done! Have a nice day ;) ------------')
logging.info('---------------------------------------------------')