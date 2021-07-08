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

    #'CH105': Animal(directory='/stelmo/mcoulter/CH105/filterframework',short_name='CH105'),

    'CH105': Animal(directory='/stelmo/david/Scn2a/coh2/CH105/AutoW/filterframework',short_name='CH105'),
    'CH112': Animal(directory='/stelmo/david/Scn2a/coh2/CH112/AutoW/filterframework',short_name='CH112'),
    'CH109': Animal(directory='/stelmo/david/Scn2a/coh2/CH109/AutoW/filterframework',short_name='CH109'),
    'CH101': Animal(directory='/stelmo/david/Scn2a/coh2/CH101/AutoW/filterframework',short_name='CH101'),

    'CH6': Animal(directory='/stelmo/mcoulter/CH6/filterframework',short_name='CH6'),
}

EDGE_ORDER = np.asarray(
    [
        (17, 16),
        (16, 15),
        (15, 13),
        (13, 12),
        (14, 13),
        (1,0),
        (1,3),
        (3,2),
        (3,5),
        (5,4),
        (5,12),
        (12,7),
        (7,6),
        (7,9),
        (9,8),
        (9,11),
        (11,10),

    ]
)

EDGE_SPACING = 15
