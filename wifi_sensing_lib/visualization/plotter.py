import numpy as np
import matplotlib.pyplot as plt
import torch

class BFRPlotter:
    """
    Visualization tool for Beamforming Reports (V-Matrices).
    Adapted from https://github.com/cheeseBG/csi-visualization.
    """
    
    @staticmethod
    def _prepare_data(v_matrix, link=(0, 0), mode='amplitude'):
        """
        Prepares V-Matrix data for plotting.
        
        Args:
            v_matrix (torch.Tensor or np.ndarray): Shape (Time, Subcarriers, Nr, Nc) or (Time, Subcarriers)
            link (tuple): (rx_idx, tx_idx) to select specific spatial stream.
            mode (str): 'amplitude' or 'phase'.
            
        Returns:
            np.ndarray: Shape (Time, Subcarriers)
        """
        if isinstance(v_matrix, torch.Tensor):
            v_matrix = v_matrix.detach().cpu().numpy()
            
        # Handle complex data
        if np.iscomplexobj(v_matrix):
            if mode == 'phase':
                data = np.angle(v_matrix)
            else:
                data = np.abs(v_matrix)
        else:
            data = v_matrix
            
        # Handle dimensions
        if data.ndim == 4: # (T, S, Nr, Nc)
            nr, nc = link
            # Check bounds
            if nr >= data.shape[2] or nc >= data.shape[3]:
                print(f"Warning: Link {link} out of bounds for shape {data.shape}. Using (0,0).")
                nr, nc = 0, 0
            data = data[:, :, nr, nc]
        elif data.ndim == 3: # (T, S, Channels) - generalized
             data = data[:, :, 0] # Take first channel if ambiguous
        
        return data

    @staticmethod
    def plot_heatmap(v_matrix, link=(0,0), mode='amplitude', title=None):
        """
        Plots a heatmap of Subcarrier vs Time.
        """
        data = BFRPlotter._prepare_data(v_matrix, link, mode)
        # data shape: (Time, Subcarriers)
        # We want X=Time, Y=Subcarriers. 
        # pcolor expects (Y, X) usually for the grid, or we transpose data.
        
        plt.figure(figsize=(12, 6))
        
        # Transpose to have Subcarriers on Y-axis, Time on X-axis
        # data.T shape: (Subcarriers, Time)
        plt.imshow(data.T, aspect='auto', cmap='jet', origin='lower')
        
        plt.colorbar(label=f'{mode.capitalize()} (dBm/rad)')
        plt.xlabel('Packet Index (Time)')
        plt.ylabel('Subcarrier Index')
        
        if title:
            plt.title(title)
        else:
            plt.title(f'BFR {mode.capitalize()} Heatmap (Link {link})')
            
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_time_series(v_matrix, subcarriers=None, link=(0,0), mode='amplitude'):
        """
        Plots time-series for specific subcarriers.
        """
        data = BFRPlotter._prepare_data(v_matrix, link, mode)
        # data: (Time, Subcarriers)
        
        if subcarriers is None:
            # Select a few evenly spaced subcarriers
            n_sub = data.shape[1]
            subcarriers = np.linspace(0, n_sub-1, 5, dtype=int)
            
        plt.figure(figsize=(12, 6))
        for sub in subcarriers:
            if sub < data.shape[1]:
                plt.plot(data[:, sub], label=f'Subcarrier {sub}', alpha=0.7)
                
        plt.xlabel('Packet Index')
        plt.ylabel(f'{mode.capitalize()}')
        plt.title(f'BFR {mode.capitalize()} Time Series (Link {link})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

