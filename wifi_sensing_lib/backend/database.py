import os
import argparse
from pathlib import Path
import torch
from tqdm import tqdm
from . import csi_backend

# ... (rest of file) ...
        # Extract V Matrix using csi_backend
        timestamps, v_matrices = csi_backend.get_v_matrix(str(pcap_path), station_addr, verbose=False)
        
        if len(timestamps) > 0:
            data = {
                "timestamps": timestamps,
                "v_matrix": torch.tensor(v_matrices, dtype=torch.complex64),
                "metadata": {
                    "room": room,
                    "subject": subject,
                    "station": station_name,
                    "address": station_addr,
                    "source_file": str(pcap_path)
                }
            }
            torch.save(data, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wi-Fi Sensing Database Extractor")
    parser.add_argument("--data_dir", required=True, help="Path to the dataset root directory (containing 'Raw')")
    args = parser.parse_args()
    
    extractor = DatabaseExtractor(args.data_dir)
    extractor.process_database()
