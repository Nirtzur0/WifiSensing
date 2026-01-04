import torch
import numpy as np

class AddComplexNoise(object):
    """
    Adds random noise to CSI data (augmentation).
    Handles both complex and real inputs (numpy or torch).
    """
    def __init__(self, ratio=0.1):
        self.ratio = ratio

    def __call__(self, csi):
        if self.ratio <= 0:
            return csi

        if isinstance(csi, np.ndarray):
            if np.iscomplexobj(csi):
                # For complex data, add noise to the magnitude
                magnitude = np.abs(csi)
                phase = np.angle(csi)
                mean_magnitude = np.mean(magnitude)
                noise_level = mean_magnitude * self.ratio
                noise = np.random.normal(0, noise_level, magnitude.shape)
                augmented_magnitude = magnitude + noise
                # Convert back to complex
                return augmented_magnitude * np.exp(1j * phase)
            else:
                # For real data, add noise directly
                mean_value = np.mean(np.abs(csi))
                noise_level = mean_value * self.ratio
                noise = np.random.normal(0, noise_level, csi.shape)
                return csi + noise

        elif isinstance(csi, torch.Tensor):
            if torch.is_complex(csi):
                # For complex tensor, add noise to the magnitude
                magnitude = torch.abs(csi)
                phase = torch.angle(csi)
                mean_magnitude = torch.mean(magnitude)
                noise_level = mean_magnitude * self.ratio
                noise = torch.randn_like(magnitude) * noise_level
                augmented_magnitude = magnitude + noise
                # Convert back to complex
                return augmented_magnitude * torch.exp(1j * phase)
            else:
                # For real tensor, add noise directly
                mean_value = torch.mean(torch.abs(csi))
                noise_level = mean_value * self.ratio
                noise = torch.randn_like(csi) * noise_level
                return csi + noise
        
        return csi
