import pandas as pd
from syntheticml.data.synthetic import Synthetic, MODELS
import os
from pylatexenc.latexencode import unicode_to_latex
import collections.abc

import sys
syn = None
DATASET_VERSION=sys.argv[1]
DATASET_VERSIONS=[f"{DATASET_VERSION}-{i+1}" for i in range(3)]
DATASET_NAME = "King County"


