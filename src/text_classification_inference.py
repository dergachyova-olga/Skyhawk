import config
import sys
import os
import logging
from text_data_loading import TextDataLoader
from text_feature_extraction import TextFeatureExtractor
from text_model import TextModel


# load configuration
default_config_folder = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'configs')
default_config_file = os.path.join(default_config_folder, 'text_classification_inference.ini')
cfg = config.load_config(sys.argv[1] if len(sys.argv) > 1 else default_config_file)


logging.info('---------------------------------------------------')
logging.info('-------- Classification module (inference) --------')
logging.info('---------------------------------------------------')


# ---------------------------- Load model --------------------------- #
classification_model = TextModel(cfg)

logging.info('Loading trained classification model...')
name, model, feature, trasformer, embeddings, label_encoder = classification_model.load()


# ---------------------------- Load data ---------------------------- #
data_loader = TextDataLoader(cfg, label_encoder)

logging.info('Loading data to be classified...')
inference_data, invalid_id = data_loader.load_classification_data('inference')


# ----------------------- Extract features -------------------------- #
feature_extractor = TextFeatureExtractor(cfg, embeddings=embeddings)

logging.info('Extracting %s features...' % feature)
inference_x = feature_extractor.get_features(feature, trasformer, inference_data)


# ------------------------- Classify ------------------------------- #
logging.info('Classifying using %s model' % name)
inference_y_ = classification_model.predict_class(model, inference_x)
labels = data_loader.decode_labels(inference_y_)


# ------------------------- Save results ------------------------------- #
logging.info('Saving classification results')

# create output folder if doesn't exist
if not os.path.exists(os.path.dirname(cfg.output.predictions_path)):
    os.mkdir(os.path.dirname(cfg.output.predictions_path))

# get correct ids for valid entries in input data
id_list = list(range(len(inference_data) + len(invalid_id)))
id_list = [i for i in id_list if i not in invalid_id]

# write predictions in a file
input_file = open(cfg.output.predictions_path, 'w')
for i, label in zip(id_list, labels):
    input_file.write("%d: %s \n" % (i, label))
input_file.close()


logging.info('---------------------------------------------------')
logging.info('------------- Done! Have a nice day ;) ------------')
logging.info('---------------------------------------------------')