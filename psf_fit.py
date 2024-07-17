#!/usr/bin/env python -W ignore::DeprecationWarning
"""
Python module for fitting PSF in cutout radio images. 

TODO:
Make this a package.
"""

__author__ = "Jaiden Cook"
__credits__ = ["Jaiden Cook"]
__version__ = "1.0.0"
__maintainer__ = "Jaiden Cook"
__email__ = "Jaiden.Cook@curtin.edu.au"

import logging
logging.captureWarnings(True) 

# Array stuff:
import numpy as np
from functions import *
from src_img import create_model_mask,footprint_mask,calc_footprint





