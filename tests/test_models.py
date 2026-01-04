import pytest
import torch
from wifi_sensing_lib.models import get_registered_models

def test_rf_crate_base_forward(rf_crate_config):
    """
    Test forward pass for rf_crate_base model.
    """
    model_name = 'rf_crate_base'
    model = get_registered_models(model_name, rf_crate_config)
    
    assert model is not None, f"Failed to instantiate {model_name}"
    
    # Dummy input: [Batch, Channels, Time, Subcarriers]
    # Check model expectation. RF_CRATE usually expects [B, C, H, W]
    batch_size = 2
    in_channels = rf_crate_config['in_channels']
    h, w = rf_crate_config['image_size']
    dummy_input = torch.randn(batch_size, in_channels, h, w, dtype=torch.complex64)
    
    output, _ = model(dummy_input)
    
    # Output shape should be [Batch, NumClasses]
    expected_output_shape = (batch_size, rf_crate_config['num_classes'])
    assert output.shape == expected_output_shape
    
    print(f"{model_name} output shape: {output.shape}")

@pytest.fixture
def rf_net_config():
    return {
        'num_classes': 6,
        'in_channels': 1,
        'seq_length': 10,
    }

def test_rf_net_forward(rf_net_config):
     """
     Test forward pass for rf_net.
     """
     model_name = 'rf_net'
     model = get_registered_models(model_name, rf_net_config)

     assert model is not None

     # rf_net expects 3D input: [Batch, Time, Dim]
     batch = 2
     t = rf_net_config['seq_length']
     dim = rf_net_config['in_channels'] # Must match in_channels (1)
     
     dummy = torch.randn(batch, t, dim)
     
     output = model(dummy)
     assert output.shape == (batch, rf_net_config['num_classes'])
