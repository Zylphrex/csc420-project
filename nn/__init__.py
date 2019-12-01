import os
import sys

crnn_path = os.path.join(
    os.getcwd(),
    'crnn.pytorch',
)

sys.path.append(crnn_path)

from .runner import extract_text
