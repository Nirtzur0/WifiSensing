import torch
from torch.utils.data import Dataset
from ..wipicap import wipicap

class WifiSensingDataset(Dataset):
    """
    Dataset wrapper for Wi-Fi Sensing data extraction using WiPiCap.
    """
    def __init__(self, pcap_file, address, config=None, transform=None):
        self.pcap_file = pcap_file
        self.address = address
        self.config = config or {}
        self.transform = transform
        
        # Extract V-Matrices
        # Returns: (timestamps, v_matrices)
        # v_matrices shape: (num_packets, num_subc, nr, nc)
        self.ts, self.vs = wipicap.get_v_matrix(pcap_file, address, verbose=True)
        self.num_samples = len(self.ts)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        v_matrix = self.vs[idx] 
        v_tensor = torch.tensor(v_matrix, dtype=torch.complex64)
        
        # Dummy labels/metadata for compatibility with RF_CRATE models
        label = 0
        user_id = 0
        orientation = 0
        rx_id = 0
        return v_tensor, torch.tensor(label), torch.tensor(user_id), torch.tensor(orientation), torch.tensor(rx_id)

class FeatureReshaper:
    """
    Reshapes and formats raw Beamforming Reports (BFR) for model consumption.
    """
    def __init__(self, config):
        self.config = config
        self.target_format = config.get('format', 'complex')
        
    def shape_convert(self, batch):
        # Input batch: [Batch, Subcarriers, Nr, Nc]
        if len(batch.shape) == 4: 
             b, s, nr, nc = batch.shape
             # Flatten Nr, Nc dimensions
             batch = batch.view(b, s, nr*nc)
             # Permute to [Batch, Channels, Time=1, Freq=Subcarriers]
             batch = batch.permute(0, 2, 1).unsqueeze(2) 
        return batch
