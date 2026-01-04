import torch
from torch.utils.data import Dataset
# Import from the internal moved copy
# Note: WiPiCap is a compiled extension. We need to insure it can be imported.
# If we moved the .so file, it should work if it's in the path.
# Since we are in wifi_sensing_lib.wipicap_adapter, and code is in wifi_sensing_lib.wipicap
from ..wipicap import wipicap # This assumes wipicap.so is importable as a module

class WiPiCapDataset(Dataset):
    def __init__(self, pcap_file, address, config, transform=None):
        self.pcap_file = pcap_file
        self.address = address
        self.config = config
        self.transform = transform
        
        self.ts, self.vs = wipicap.get_v_matrix(pcap_file, address, verbose=True)
        self.num_samples = len(self.ts)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        v_matrix = self.vs[idx] 
        v_tensor = torch.tensor(v_matrix, dtype=torch.complex64)
        label = 0
        user_id = 0
        orientation = 0
        rx_id = 0
        return v_tensor, torch.tensor(label), torch.tensor(user_id), torch.tensor(orientation), torch.tensor(rx_id)

class WiPiCapDataShapeConverter:
    def __init__(self, config):
        self.config = config
        self.target_format = config.get('format', 'complex')
        
    def shape_convert(self, batch):
        if len(batch.shape) == 4: 
             b, s, nr, nc = batch.shape
             batch = batch.view(b, s, nr*nc)
             batch = batch.permute(0, 2, 1).unsqueeze(2) 
        return batch
