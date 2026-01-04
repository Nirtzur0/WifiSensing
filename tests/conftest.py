import pytest
import os
import torch
import sys

# Ensure wifi_sensing_lib is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(scope="session")
def pcap_path():
    """
    Returns the absolute path to the local PCAP file for testing.
    """
    # Adjust this path if necessary to match the actual location
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, "wifi_sensing_lib", "data", "local_pcaps", "L_62_A_12_c1_n_1_AP_4x4-2.pcapng")
    
    if not os.path.exists(path):
        pytest.skip(f"PCAP file not found at {path}")
    return path

@pytest.fixture
def basic_config():
    """
    Returns a basic configuration dictionary for models and datasets.
    """
    return {
        'seq_length': 10,  # Number of frames per window
        'stride': 5,
        'format': 'complex', # Default format
        'model_input_shape': 'BCHW-C',
        'samp_rate': 1000,
        'window_size': 256,
        'window_step': 30,
        # Model specific
        'num_classes': 6,
        'in_channels': 1, # Depends on format
    }

@pytest.fixture
def station_address():
    # This might need to be adjusted based on the specific PCAP content
    # Using the one from previous context or a common one found in that file
    # If uncertain, we can try to find one dynamically or use a known one.
    # From context: L_62_A_12_c1_n_1_AP_4x4-2.pcapng was used in CLI with default or explicit address.
    # Let's use a broadcast or specific if known. 
    # Previous CLI command context didn't show explicit address, so maybe it matters.
    # Let's use wildcard or a known one if we can discover it.
    # For now, let's assume we can filter by the AP-Client link.
    # Re-using the logic from pcap_dataset which might need an specific address.
    # If unknown, we can return None and tests can handle it or use a default.
    return "ff:ff:ff:ff:ff:ff" # Broadcast/All for now

@pytest.fixture
def rf_crate_config():
    return {
        'num_classes': 6,
        'in_channels': 3, # Corrected from 1 to 3 to match reshaper output for Nr=3
        'image_size': [10, 64], # Time, Subcarriers
        'patch_size': [2, 8],
        'feedforward': 'type1', # Corrected from 'crate'
        'relu_type': 'crelu', # Corrected from 'gelu'
        'patch_embedding_method': 'conv_patch', # Corrected
        'mlp_head_type': 'crate_version', # Corrected from 'linear'
        'output_type': 'cls',
    }
