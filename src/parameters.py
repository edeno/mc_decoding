from os.path import abspath, dirname, join, pardir

import numpy as np
from loren_frank_data_processing import Animal

# LFP sampling frequency
SAMPLING_FREQUENCY = 500

# Data directories and definitions
ROOT_DIR = join(abspath(dirname(__file__)), pardir)
RAW_DATA_DIR = join(ROOT_DIR, 'Raw-Data')
PROCESSED_DATA_DIR = join(ROOT_DIR, 'Processed-Data')
FIGURE_DIR = join(ROOT_DIR, 'figures')

ANIMALS = {
    'david905': Animal(directory='/stelmo/mcoulter/david905/filterframework',
                       short_name='david905'),
}

EDGE_ORDER = np.asarray(
    [
        (17, 16),
        (16, 15),
        (15, 13),
        (13, 12),
        (14, 13),
        (12, 5),
        (5, 4),
        (5, 3),
        (3, 2),
        (3, 1),
        (1, 0),
        (12, 7),
        (7, 6),
        (7, 9),
        (9, 8),
        (9, 11),
        (11, 10),
    ]
)

EDGE_SPACING = 15
