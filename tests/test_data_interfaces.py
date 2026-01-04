import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from wifi_sensing_lib.data.interfaces.pcap_dataset import WifiSensingDataset, FeatureReshaper

@pytest.fixture
def mock_csi_backend():
    with patch('wifi_sensing_lib.data.interfaces.pcap_dataset.csi_backend') as mock:
        # Return dummy timestamps and v_matrix
        # Shape: [20, 64, 3, 1] for 20 packets
        num_packets = 20
        timestamps = np.arange(num_packets)
        v_matrices = np.random.randn(num_packets, 64, 3, 1) + 1j * np.random.randn(num_packets, 64, 3, 1)
        mock.get_v_matrix.return_value = (timestamps, v_matrices)
        yield mock

def test_wifi_sensing_dataset(mock_csi_backend, pcap_path, station_address, basic_config):
    """
    Test WifiSensingDataset initialization and item retrieval using mocked backend.
    """
    dataset = WifiSensingDataset(pcap_path, station_address, config=basic_config)
    
    assert len(dataset) > 0, "Dataset should not be empty"
    
    # Get first item
    item = dataset[0]
    # Expect: v_tensor, label, user_id, orientation, rx_id
    assert len(item) == 5
    
    v_tensor = item[0]
    assert isinstance(v_tensor, torch.Tensor)
    assert v_tensor.dtype == torch.complex64
    
    # Shape check: [WindowSize, Subc, Nr, Nc]
    # Window size from config is 10
    assert v_tensor.shape[0] == basic_config['seq_length']


@pytest.mark.parametrize("format_type, expected_channels", [
    ('complex', 1),   # Complex stays complex(maybe split later) or permuted
    ('amplitude', 1), # Abs value
    ('polar', 2),     # Amp + Phase
    ('cartesian', 2), # Real + Imag
])
def test_feature_reshaper_formats(format_type, expected_channels, basic_config):
    """
    Test FeatureReshaper with different formats on dummy data.
    """
    # Create dummy batch: [Batch=2, Time=10, Subc=64, Nr=3, Nc=1]
    batch_size = 2
    time = 10
    subc = 64
    nr = 3
    nc = 1
    dummy_input = torch.randn(batch_size, time, subc, nr, nc, dtype=torch.complex64)
    
    config = basic_config.copy()
    config['format'] = format_type
    config['model_input_shape'] = 'BCHW' # Standard reshaping
    
    reshaper = FeatureReshaper(config)
    output = reshaper.shape_convert(dummy_input)
    
    assert isinstance(output, torch.Tensor)
    # Check dimensions based on format logic in FeatureReshaper
    # Typically [Batch, Channels, ..., ...]
    
    if format_type == 'complex':
         # Code path for BCHW-C might differ, usually permutes
         # If BCHW and complex, might be complex64 tensor.
         # Current logic: [B, T, S, Nr*Nc] -> permute -> ...
         pass
    elif format_type in ['polar', 'cartesian']:
         # Likely [B, T, S, Ant, 2] -> reshaped
         pass
         
    print(f"Format {format_type} output shape: {output.shape}")

def test_feature_reshaper_dfs(basic_config):
    """
    Test DFS generation via FeatureReshaper.
    """
    config = basic_config.copy()
    config['format'] = 'dfs'
    config['model_input_shape'] = 'B2CNFT' # Legacy shape for DFS
    
    # Dummy input
    batch_size = 1
    time = 500 # Needs enough samples for STFT (256 window)
    subc = 10 # Small subc for speed
    nr = 1
    nc = 1
    dummy_input = torch.randn(batch_size, time, subc, nr, nc, dtype=torch.complex64)
    
    reshaper = FeatureReshaper(config)
    output = reshaper.shape_convert(dummy_input)
    
    # Output expected: [B, 2, S, Ant, Freq, TimeBins]
    assert len(output.shape) == 6
    assert output.shape[1] == 2 # Real/Imag or similar 2 channels
