import os
import configparser
from collections import namedtuple
from collections import OrderedDict
import ast
import logging
import tensorflow as tf


# make this run deterministic
import random
import numpy
random.seed(1)
numpy.random.seed(1)


def load_config(config_path):

    # load configuration file
    config = configparser.ConfigParser()
    config.read(config_path)

    # convert config sections and options into objects
    undefined = list()
    config_sections = OrderedDict()
    for section in config.sections():

        # create dict of section options
        section_options = OrderedDict()
        for option in config.options(section):
            if config.get(section, option) == '':
                undefined.append(section + '.' + option)
                section_options[option] = None
            else:
                section_options[option] = typed(config.get(section, option))

        # put section dict to config dict
        section_struct = namedtuple(section, ' '.join([option for option in config.options(section)]))
        config_sections[section] = section_struct(**section_options)

    # create config object
    config_struct = namedtuple('config', ' '.join([key for key in config_sections.keys()]))
    config = config_struct(**config_sections)

    # set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    # mute internal TensorFlow messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.getLogger('tensorflow').disabled = True
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.get_logger().setLevel('INFO')

    return config


def typed(value):
    try:
        value = ast.literal_eval(value)
        return value
    except:
        return value
