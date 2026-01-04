import pytest
import numpy as np
from wifi_sensing_lib.backend import csi_backend

def test_get_v_matrix(pcap_path, station_address):
    """
    Test extraction of V-Matrices from a real PCAP file.
    """
    print(f"Testing extraction from: {pcap_path}")
    
    # Run extraction
    timestamps, v_matrices = csi_backend.get_v_matrix(pcap_path, station_address, verbose=False)
    
    # Assertions
    assert isinstance(timestamps, np.ndarray), "Timestamps should be a numpy array"
    assert isinstance(v_matrices, np.ndarray), "V-Matrices should be a numpy array"
    
    assert len(timestamps) == len(v_matrices), "Number of timestamps and V-matrices must match"
    
    if len(v_matrices) > 0:
        # Check shape: [NumPackets, Subcarriers, Nr, Nc]
        shape = v_matrices.shape
        assert len(shape) == 4, f"Expected 4 dimensions (N, Subc, Nr, Nc), got {shape}"
        
        # Typical values specific to this dataset might be known, e.g., 64 subcarriers usually?
        # Or specific Nr/Nc. We can assert they are non-zero.
        assert shape[1] > 0, "Number of subcarriers should be positive"
        assert shape[2] > 0, "Number of RX antennas should be positive"
        assert shape[3] > 0, "Number of TX antennas should be positive"
        
        print(f"Extracted {len(v_matrices)} packets. Shape: {shape}")
    else:
        pytest.main.fail("No packets extracted from the PCAP file. Check filters or file content.")
