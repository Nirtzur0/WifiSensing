import pytest
import os
import torch
import tempfile
import yaml
from unittest.mock import patch, MagicMock
from wifi_sensing_lib.inference.pipeline import InferencePipeline

# Need a dummy config file for the pipeline
@pytest.fixture
def pipeline_config_path(rf_crate_config):
    # Create a temporary config yaml (flat structure)
    config = {
        'model_name': 'rf_crate_base',
        **rf_crate_config, # Merge model params
        # Dataset params
        'seq_length': 10,
        'stride': 5,
        'format': 'complex', 
        'model_input_shape': 'BCHW-C',
        # Inference params
        'batch_size': 2,
        'device': 'cpu'
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
        yaml.dump(config, tmp)
        path = tmp.name
        
    yield path
    
    os.remove(path)

def test_inference_pipeline_run(pipeline_config_path, pcap_path, station_address):
    """
    Test running the full inference pipeline on a PCAP file with mocked dataset.
    """
    print(f"Initializing pipeline with config: {pipeline_config_path}")
    
    # Mock WifiSensingDataset to avoid reading PCAP
    with patch('wifi_sensing_lib.inference.pipeline.WifiSensingDataset') as MockDataset:
        # Configure output of dataset
        # Dataset returns: v_tensor, label, user_id, orientation, rx_id
        # v_tensor shape: [10, 64, 3, 1] (Seq, S, Nr, Nc) -> Complex64
        mock_instance = MockDataset.return_value
        mock_instance.__len__.return_value = 5 # 5 samples
        
        # Create dummy item
        dummy_v = torch.randn(10, 64, 3, 1, dtype=torch.complex64)
        dummy_label = torch.tensor(0)
        dummy_user = torch.tensor(1)
        dummy_ori = torch.tensor(1)
        dummy_rx = torch.tensor(0)
        
        mock_instance.__getitem__.return_value = (dummy_v, dummy_label, dummy_user, dummy_ori, dummy_rx)
        
        pipeline = InferencePipeline(pipeline_config_path)
        
        # Run pipeline
        try:
            pipeline.run(pcap_path, station_address=station_address)
            
            print("Pipeline run completed.")
            
        except Exception as e:
            pytest.fail(f"Pipeline run failed: {e}")
