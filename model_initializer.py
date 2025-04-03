import sys
import os

# Pathleri PYTHONPATH'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "ckpts/schp")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "ckpts/densepose")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "preprocess")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "preprocess/humanparsing")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "preprocess/openpose")))

from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
from leffa_utils.densepose_predictor import DensePosePredictor
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose

# Bu dosya import edildiğinde otomatik olarak yol tanımlamaları yapılmış olacak.
