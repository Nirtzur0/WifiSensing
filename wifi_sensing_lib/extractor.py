import os
import argparse
from pathlib import Path
import torch
from tqdm import tqdm
from .wipicap import wipicap

# Default Configuration
DEFAULT_STATIONS = {
    "9C": "B0:B9:8A:63:55:9C",
    "25": "38:94:ED:12:3C:25",
    "89": "CC:40:D0:57:EA:89"
}

DEFAULT_ROOMS = ["Classroom", "Kitchen", "Livingroom"]

class DatabaseExtractor:
    """
    Extracts Wi-Fi sensing data (V-Matrices) from a structured raw dataset.
    """
    def __init__(self, root_dir, stations=None, rooms=None):
        self.root_dir = Path(root_dir)
        self.raw_dir = self.root_dir / "Raw"
        self.processed_dir = self.root_dir / "Processed"
        self.stations = stations or DEFAULT_STATIONS
        self.rooms = rooms or DEFAULT_ROOMS

    def process_database(self):
        """
        Traverses: Raw/<Room>/<Subject>/<file.pcap>
        Outputs: Processed/<Room>/<Station>/vtilde_matrices/<file>_<station>.pt
        """
        if not self.raw_dir.exists():
            print(f"Error: Raw directory not found at {self.raw_dir}")
            return

        print(f"Processing database at {self.root_dir}")
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        for room in self.rooms:
            self._process_room(room)

    def _process_room(self, room):
        room_path = self.raw_dir / room
        if not room_path.exists():
            # print(f"Skipping {room} (not found)")
            return
        
        # Iterate over subjects (subdirectories in Room)
        for subject_path in room_path.iterdir():
            if not subject_path.is_dir():
                continue
            
            print(f"  Processing Subject: {subject_path.name} in {room}")
            pcap_files = list(subject_path.glob("*.pcap")) + list(subject_path.glob("*.pcapng"))
            
            for pcap_file in tqdm(pcap_files, desc=f"    Extracting"):
                self._process_file_for_all_stations(pcap_file, room, subject_path.name)

    def _process_file_for_all_stations(self, pcap_path, room, subject):
        basename = pcap_path.stem

        for station_name, station_addr in self.stations.items():
            try:
                self._extract_station_data(pcap_path, room, subject, station_name, station_addr, basename)
            except Exception as e:
                print(f"Error processing {pcap_path.name} for {station_name}: {e}")

    def _extract_station_data(self, pcap_path, room, subject, station_name, station_addr, basename):
        # Define output path
        output_dir = self.processed_dir / room / station_name / "vtilde_matrices"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{basename}_{station_name}.pt"
        
        if output_file.exists():
            return

        # Extract V Matrix using WiPiCap
        timestamps, v_matrices = wipicap.get_v_matrix(str(pcap_path), station_addr, verbose=False)
        
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
