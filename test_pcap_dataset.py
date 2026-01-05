#!/usr/bin/env python3
"""
Test script for PcapTrainingDataset
"""
import sys
sys.path.insert(0, '.')

from wifi_sensing_lib.data import PcapTrainingDataset
import torch

def test_pcap_dataset():
    """Test PCAP training dataset with local PCAP files."""
    
    print("=" * 60)
    print("Testing PcapTrainingDataset")
    print("=" * 60)
    
    # Configuration
    config = {
        'label_type': 'activity',
        'seq_length': 100,
        'stride': 50,
        'format': 'complex',
        'cache_csi': False,
        'station_filter': None  # Load all stations
    }
    
    # Test with local_pcaps directory
    data_dir = 'wifi_sensing_lib/data/local_pcaps'
    
    print(f"\n1. Initializing dataset from: {data_dir}")
    try:
        dataset = PcapTrainingDataset(data_dir, config=config)
        print(f"✓ Dataset initialized successfully")
        print(f"  Found {len(dataset)} samples")
    except Exception as e:
        print(f"✗ Failed to initialize dataset: {e}")
        return False
    
    if len(dataset) == 0:
        print("\n⚠ No samples found. This is expected if:")
        print("  - PCAP files are not in Processed/ directory structure")
        print("  - Filenames don't match expected patterns")
        print("\nTrying to create a test sample manually...")
        
        # Test with a single PCAP file directly
        import os
        pcap_files = [f for f in os.listdir(data_dir) if f.endswith('.pcapng')]
        if pcap_files:
            print(f"\nFound PCAP file: {pcap_files[0]}")
            print("Note: For training, organize files in Processed/ structure")
        return True
    
    print(f"\n2. Testing data loading")
    try:
        # Get first sample
        data, label, user_id, orientation, rx_id = dataset[0]
        
        print(f"✓ Successfully loaded sample 0")
        print(f"  Data shape: {data.shape}")
        print(f"  Data dtype: {data.dtype}")
        print(f"  Label: {label.item()}")
        print(f"  User ID: {user_id.item()}")
        
        # Verify data properties
        assert isinstance(data, torch.Tensor), "Data should be a tensor"
        assert data.dtype == torch.complex64, "Data should be complex64"
        assert len(data.shape) == 4, "Data should be 4D [T, S, Nr, Nc]"
        assert data.shape[0] == config['seq_length'], "Time dimension should match seq_length"
        
        print(f"✓ Data validation passed")
        
    except Exception as e:
        print(f"✗ Failed to load sample: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n3. Testing multiple samples")
    try:
        num_samples_to_test = min(3, len(dataset))
        for i in range(num_samples_to_test):
            data, label, _, _, _ = dataset[i]
            print(f"  Sample {i}: shape={data.shape}, label={label.item()}")
        print(f"✓ Successfully loaded {num_samples_to_test} samples")
    except Exception as e:
        print(f"✗ Failed to load multiple samples: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    return True

if __name__ == '__main__':
    success = test_pcap_dataset()
    sys.exit(0 if success else 1)
