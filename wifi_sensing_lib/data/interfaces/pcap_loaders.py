import torch
from torch.utils.data import DataLoader
import numpy as np

def pcap_collate_fn(batch):
    """
    Custom collate function to convert a batch of numpy arrays to tensors.
    """
    # Separate the components of the batch
    v_windows, labels, station_labels, orientations, rx_ids = zip(*batch)

    # Stack the numpy arrays to create batch tensors
    v_tensor = torch.tensor(np.array(v_windows), dtype=torch.complex64)
    labels_tensor = torch.tensor(np.array(labels), dtype=torch.long)
    station_labels_tensor = torch.tensor(np.array(station_labels), dtype=torch.long)
    orientations_tensor = torch.tensor(np.array(orientations), dtype=torch.long)
    rx_ids_tensor = torch.tensor(np.array(rx_ids), dtype=torch.long)

    return (
        v_tensor,
        labels_tensor,
        station_labels_tensor,
        orientations_tensor,
        rx_ids_tensor
    )

def make_pcap_dataloader(dataset, is_training, generator, batch_size, collate_fn_padd=None, num_workers=0):
    """
    Creates a DataLoader for the PCAP dataset.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        generator=generator,
        num_workers=num_workers,
        collate_fn=pcap_collate_fn
    )

def pcap_data_shape_converter(config):
    """
    Dummy function to satisfy the trainer.
    """
    pass
