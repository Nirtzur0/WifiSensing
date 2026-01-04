import os
import glob
import argparse
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

# Import internal modules
from .wipicap import wipicap

# Configuration matching the bash script
STATIONS = {
    "9C": "B0:B9:8A:63:55:9C",
    "25": "38:94:ED:12:3C:25",
    "89": "CC:40:D0:57:EA:89"
}

ROOMS = ["Classroom", "Kitchen", "Livingroom"]

class DatabaseExtractor:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.raw_dir = self.root_dir / "Raw"
        self.processed_dir = self.root_dir / "Processed"

    def process_database(self):
        """
        Traverses the database structure and processes each PCAP file.
        Structure expected: Raw/<Room>/<Subject>/<file.pcap>
        Output: Processed/<Room>/<Station>/vtilde_matrices/<file>_<station>.pt
        """
        if not self.raw_dir.exists():
            print(f"Error: Raw directory not found at {self.raw_dir}")
            return

        print(f"Processing database at {self.root_dir}")
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        for room in ROOMS:
            room_path = self.raw_dir / room
            if not room_path.exists():
                print(f"Skipping {room} (not found)")
                continue
            
            # Iterate over subjects (subdirectories in Room)
            for subject_path in room_path.iterdir():
                if not subject_path.is_dir():
                    continue
                
                print(f"  Processing Subject: {subject_path.name}")
                pcap_files = list(subject_path.glob("*.pcap")) + list(subject_path.glob("*.pcapng"))
                
                for pcap_file in tqdm(pcap_files, desc=f"    Files in {subject_path.name}"):
                    self._process_file(pcap_file, room, subject_path.name)

    def _process_file(self, pcap_path, room, subject):
        basename = pcap_path.stem

        for station_name, station_addr in STATIONS.items():
            # Define output path
            # Structure: Processed/<Room>/<Station>/vtilde_matrices/
            output_dir = self.processed_dir / room / station_name / "vtilde_matrices"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / f"{basename}_{station_name}.pt"
            
            if output_file.exists():
                # Skip if already exists? Or overwrite? 
                # For now let's skip to save time on re-runs
                continue

            try:
                # Extract V Matrix using WiPiCap
                # verbose=False to keep tqdm clean
                timestamps, v_matrices = wipicap.get_v_matrix(str(pcap_path), station_addr, verbose=False)
                
                if len(timestamps) > 0:
                    # Save as PyTorch tensors
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
                    # print(f"      Saved {len(timestamps)} packets for {station_name}")
            except Exception as e:
                print(f"Error processing {pcap_path.name} for {station_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wi-Fi Sensing Database Extractor")
    parser.add_argument("--data_dir", required=True, help="Path to the 'Data/DatasetName' directory containing 'Raw'")
    args = parser.parse_args()
    
    extractor = DatabaseExtractor(args.data_dir)
    extractor.process_database()
