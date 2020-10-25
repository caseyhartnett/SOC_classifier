import sys
import os
from soc_classifier import SocClassifier

CURR_PATH = os.path.dirname(os.path.realpath(__file__))

soc_code_model = SocClassifier('soc_code')
soc_code_model.pickle_pipeline(os.path.join(CURR_PATH, '../models/soc_code_model.pickle'))

soc_major_model = SocClassifier('major_group')
soc_major_model.pickle_pipeline(os.path.join(CURR_PATH, '../models/soc_major_model.pickle'))

soc_high_level_model = SocClassifier('high_level_groups')
soc_high_level_model.pickle_pipeline(os.path.join(CURR_PATH, '../models/soc_high_level_model.pickle'))
