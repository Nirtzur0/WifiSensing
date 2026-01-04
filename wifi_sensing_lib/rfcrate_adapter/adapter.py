from ..rfcrate.Models import *
from ..rfcrate.Datasets import *
from ..rfcrate.func_utils import *

def load_model(model_name, config, device):
    """Safe wrapper to load RF_CRATE model"""
    # Note: RF_CRATE internal registry uses strings. 
    # We rely on RF_CRATE's Models.__init__ to have populated registered_models
    from ..rfcrate.Models import get_registered_models
    return get_registered_models(model_name, config).to(device)
