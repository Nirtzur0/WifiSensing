from . import models
from . import data
from . import backend
from . import inference
from . import training

# Expose key components directly for convenience
from .models import get_registered_models, RF_CRATE, RFNet
from .data import WifiSensingDataset, FeatureReshaper
from .inference import InferencePipeline
from .training import run_experiment
from .backend import csi_backend
