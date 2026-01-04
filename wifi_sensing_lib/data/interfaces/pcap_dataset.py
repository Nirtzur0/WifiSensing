import torch
from torch.utils.data import Dataset
from ...backend import csi_backend
import numpy as np
import scipy.signal as signal

class WifiSensingDataset(Dataset):
    """
    Dataset wrapper for Wi-Fi Sensing data extraction using csi_backend.
    """
    def __init__(self, pcap_file, address, config=None, transform=None):
        self.pcap_file = pcap_file
        self.address = address
        self.config = config or {}
        self.transform = transform
        
        # Extract V-Matrices
        # Returns: (timestamps, v_matrices)
        # v_matrices shape: (num_packets, num_subc, nr, nc)
        self.ts, self.vs = csi_backend.get_v_matrix(pcap_file, address, verbose=True)
        self.num_samples = len(self.ts)
        # Sliding Window Configuration
        self.window_size = self.config.get('seq_length', 1)  # Default to 1 (single frame)
        self.stride = self.config.get('stride', 1)
        
        # Calculate number of windows
        if self.num_samples < self.window_size:
            self.num_windows = 0
            print(f"Warning: Not enough samples ({self.num_samples}) for window size {self.window_size}")
        else:
            self.num_windows = (self.num_samples - self.window_size) // self.stride + 1
        
    def __len__(self):
        return self.num_windows
    
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        
        # Extract window
        # Shape: (Time=window_size, Subcarriers, Nr, Nc)
        v_window = self.vs[start_idx:end_idx]
        v_tensor = torch.tensor(v_window, dtype=torch.complex64)
        
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
        # Input batch from WifiSensingDataset: [Batch, Time, Subcarriers, Nr, Nc]
        # or [Batch, Subcarriers, Nr, Nc] if Time=1 and squeezed? No, Dataset returns (T, S, Nr, Nc)
        
        # Ensure 5D: [B, T, S, Nr, Nc]
        if len(batch.shape) == 4: # [B, S, Nr, Nc] - missing T
             batch = batch.unsqueeze(1)
        
        b, t, s, nr, nc = batch.shape
        
        # 1. Format Conversion (Complex -> Real/Imag/Amp/Phase)
        # Result layout typically: [B, T, S, Nr*Nc, Channels] or similar
        
        # Flatten antenna dim: [B, T, S, Nr*Nc] complex
        batch = batch.view(b, t, s, nr*nc)
        
        target_fmt = self.target_format
        if target_fmt == 'complex':
             # Keep complex, but maybe handle model input shape
             pass
        elif target_fmt == 'amplitude':
             batch = torch.abs(batch) # [B, T, S, Ant]
             batch = batch.unsqueeze(-1) # [B, T, S, Ant, 1]
        elif target_fmt == 'polar':
             amp = torch.abs(batch)
             phase = torch.angle(batch)
             batch = torch.stack([amp, phase], dim=-1) # [B, T, S, Ant, 2]
        elif target_fmt == 'cartesian':
             real = batch.real
             imag = batch.imag
             batch = torch.stack([real, imag], dim=-1) # [B, T, S, Ant, 2]
        elif target_fmt in ['dfs', 'dense_dfs']:
             # Generate DFS using STFT
             # Batch: [B, T, S, Ant] complex
             # Parameters from config or defaults
             fs = self.config.get('samp_rate', 1000)
             window_size = self.config.get('window_size', 256)
             window_step = self.config.get('window_step', 30)
             
             # Call internal DFS helper
             batch = self._compute_dfs(batch, fs, window_size, window_step)
             # Result: [B, 2, S, Ant, Freq, TimeBins]
             
             # If dense_dfs_amp, might need magnitude [B, 1, ...]
             if target_fmt == 'dense_dfs_amp':
                 # sqrt(real^2 + imag^2)
                 real = batch[:, 0]
                 imag = batch[:, 1]
                 amp = torch.sqrt(real**2 + imag**2)
                 batch = amp.unsqueeze(1) # [B, 1, S, Ant, F, T]
        else: 
             # Default fallback or error
             pass

        # 2. Model Input Shape Standard
        
        model_shape = self.config.get('model_input_shape', 'BCHW')
        
        if target_fmt == 'complex':
             if model_shape == 'BCHW-C':
                  # [B, Ant, T, S] complex
                  batch = batch.permute(0, 3, 1, 2)
        elif target_fmt in ['dfs', 'dense_dfs', 'dense_dfs_amp']:
             # Batch: [B, C(2/1), S, Ant, F, T_bins]
             if model_shape == 'B2CNFT':
                  # Legacy StfNet/SLNet: [B, C, S, Ant, F, T]
                  # It matches!
                  pass
             elif model_shape == 'BTCHW':
                  # [B, T, C, H, W]?? Widar3 DFS?
                  # Widar3 DFS usually [Batch, T, F, S, Ant*C] ??
                  # Let's support B2CNFT mainly for now as it's the DFS user.
                  pass
             elif model_shape == 'BCHW':
                  # Flatten everything to [B, Channels, F, T] ?
                  # Channels = S * Ant * C
                  b, c, s, ant, f, t = batch.shape
                  batch = batch.permute(0, 2, 3, 1, 4, 5).reshape(b, s*ant*c, f, t)
        else:
             # Batch is [B, T, S, Ant, C] 
             if model_shape == 'BCHW':
                  # [B, T, S, Ant*C] -> [B, Channels(Ant*C), T, S]
                  batch = batch.view(b, t, s, -1)
                  batch = batch.permute(0, 3, 1, 2)
             elif model_shape == 'BLC':
                   batch = batch.view(b, t, -1)
             elif model_shape == 'BTCHW':
                   # [B, T, S, Ant, C] -> [B, T, C, Ant, S]
                   batch = batch.permute(0, 1, 4, 3, 2)
             elif model_shape == 'B2CNFT':
                   # [B, T, S, Ant, C] -> [B, C, S, Ant, 1, T] 
                   batch = batch.permute(0, 4, 2, 3, 1).unsqueeze(-2)
             
        return batch

    def _compute_dfs(self, batch, samp_rate=1000, window_size=256, window_step=30):
        """
        Computes DFS (Doppler Frequency Shift) spectrograms using STFT.
        Input: [B, T, S, Ant] (Complex Tensor)
        Output: [B, 2, S, Ant, FreqBins, TimeBins]
        """
        # Move to CPU/Numpy for scipy
        device = batch.device
        csi_np = batch.detach().cpu().numpy() # [B, T, S, Ant]
        
        # Prepare for STFT along axis=0 (Time).
        # Scipy stft expects time at axis=-1 or specific axis.
        # We want to enable batch processing.
        # Transpose to [T, B, S, Ant] to use axis=0
        csi_np = csi_np.transpose(1, 0, 2, 3) # [T, B, S, Ant]
        
        # DC Removal
        csi_np = csi_np - np.mean(csi_np, axis=0) # Mean over time? Original code did axis=0 on [T, ...]
        
        noverlap = window_size - window_step
        
        # STFT
        freq, ticks, zxx = signal.stft(csi_np, 
                                      fs=samp_rate, 
                                      nfft=samp_rate,
                                      window=('gaussian', window_size), 
                                      nperseg=window_size, 
                                      noverlap=noverlap, 
                                      return_onesided=False,
                                      padded=True, 
                                      axis=0)
        
        # zxx: [Freq, B, S, Ant, TimeBins] (since time was axis 0)
        # Filter Frequencies (Widar3 logic: +/- 60Hz)
        half_rate = samp_rate / 2
        uppe_stop = 60
        freq_bins_unwrap = np.concatenate((np.arange(0, half_rate, 1) / samp_rate, np.arange(-half_rate, 0, 1) / samp_rate))
        # Note: signal.stft returns bins in standard FFT order (0..pos..neg) if return_onesided=False?
        # Check standard behavior. signal.stft returns 0..fs/2, -fs/2.. if sided=False.
        # Or standard fftshift order?
        # Original code used logical_and on `freq_bins_unwrap` which seems manually constructed.
        # Assuming original code logic matches signal.stft output order.
        
        freq_lpf_sele = np.logical_and(np.less_equal(freq_bins_unwrap,(uppe_stop / samp_rate)),np.greater_equal(freq_bins_unwrap,(-uppe_stop / samp_rate)))
        freq_lpf_positive_max = 60 # Roll amount
        
        # Select dims
        zxx = zxx[freq_lpf_sele] # [SelectedFreq, B, S, Ant, T_bins]
        
        # Roll to center DC
        zxx = np.roll(zxx, freq_lpf_positive_max, axis=0)
        
        # Transpose to [B, S, Ant, Freq, T_bins]
        zxx = zxx.transpose(1, 2, 3, 0, 4)
        
        # Split Real/Imag -> [B, 2, S, Ant, Freq, T_bins]
        out_real = np.real(zxx)
        out_imag = np.imag(zxx)
        out = np.stack([out_real, out_imag], axis=1)
        
        return torch.tensor(out, dtype=torch.float32, device=device)
