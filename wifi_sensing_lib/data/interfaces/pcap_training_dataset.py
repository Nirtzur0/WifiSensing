import os
import re
import torch
from torch.utils.data import Dataset
from pathlib import Path
from ...backend import csi_backend
import numpy as np


class PcapTrainingDataset(Dataset):
    """
    Dataset for training on PCAP files with automatic label extraction.
    
    Supports multiple labeling schemes:
    1. Directory structure: Data/{Dataset}/Processed/{Room}/{Station}/FeedBack_Pcap/{file}.pcapng
    2. Filename patterns: L_{room}_A_{activity}_c{config}_n_{num}_AP_{antenna}.pcapng
    
    Args:
        data_dir: Root directory containing PCAP files
        config: Configuration dict with:
            - label_type: 'activity', 'room', or 'station'
            - seq_length: Window size for temporal sequences
            - stride: Stride for sliding window
            - format: Data format (complex, amplitude, etc.)
            - station_filter: Optional list of stations to include
            - cache_csi: Whether to cache extracted CSI data
    """
    
    def __init__(self, data_dir, config=None, transform=None):
        # Handle case where Trainer passes config dict as first argument
        if isinstance(data_dir, dict):
            config = data_dir
            # specific key used in run_pipeline/configs is 'dataset_path'
            target_path = config.get('dataset_path', config.get('data_dir', 'Data')) 
            data_dir = target_path
            
        self.data_dir = Path(data_dir)
        self.config = config or {}
        self.transform = transform
        
        self.label_type = self.config.get('label_type', 'activity')
        self.seq_length = self.config.get('seq_length', 100)
        self.stride = self.config.get('stride', 50)
        self.station_filter = self.config.get('station_filter', None)
        self.cache_csi = self.config.get('cache_csi', False)
        
        # Station MAC address mapping
        self.station_map = {
            '9C': 'B0:B9:8A:63:55:9C',
            '25': '38:94:ED:12:3C:25',
            '89': 'CC:40:D0:57:EA:89'
        }
        
        # Activity labels A-U
        self.activity_labels = list('ABCDEFGHIJKLMNOPQRSTU')
        self.room_labels = ['Classroom', 'Kitchen', 'Livingroom']
        
        # Scan and index all PCAP files
        self.samples = []
        self._scan_directory()
        
        print(f"Found {len(self.samples)} PCAP files")
        
        # Cache for CSI data if enabled
        self.csi_cache = {} if self.cache_csi else None
        
    def _scan_directory(self):
        """Scan directory structure to find and label PCAP files."""
        
        # Pattern 1: Processed directory structure
        processed_dir = self.data_dir / 'Processed'
        if processed_dir.exists():
            for room_dir in processed_dir.iterdir():
                if not room_dir.is_dir():
                    continue
                room = room_dir.name
                
                for station_dir in room_dir.iterdir():
                    if not station_dir.is_dir():
                        continue
                    station = station_dir.name
                    
                    # Filter by station if specified
                    if self.station_filter and station not in self.station_filter:
                        continue
                    
                    # Look in FeedBack_Pcap subdirectory
                    pcap_dir = station_dir / 'FeedBack_Pcap'
                    if pcap_dir.exists():
                        for pcap_file in pcap_dir.glob('*.pcapng'):
                            label_info = self._extract_labels_from_path(
                                pcap_file, room, station
                            )
                            if label_info:
                                self.samples.append(label_info)
        
        # Pattern 2: Flat directory with labeled filenames
        for pcap_file in self.data_dir.rglob('*.pcapng'):
            if 'Processed' in str(pcap_file):
                continue  # Already handled above
            
            label_info = self._extract_labels_from_filename(pcap_file)
            if label_info:
                self.samples.append(label_info)
    
    def _extract_labels_from_path(self, pcap_file, room, station):
        """Extract labels from directory structure and filename."""
        
        # Get MAC address for station
        mac_address = self.station_map.get(station, 'ff:ff:ff:ff:ff:ff')
        
        # Try to extract activity from filename
        # Pattern: {prefix}_{activity}_{suffix}.pcapng
        filename = pcap_file.stem
        activity = None
        
        # Look for activity letter in filename
        for label in self.activity_labels:
            if f'_{label}_' in filename or filename.startswith(f'{label}_'):
                activity = label
                break
        
        # If no activity found, try pattern L_XX_A_YY
        if not activity:
            match = re.search(r'_([A-U])_', filename)
            if match:
                activity = match.group(1)
        
        return {
            'pcap_file': str(pcap_file),
            'mac_address': mac_address,
            'room': room,
            'station': station,
            'activity': activity,
            'room_label': self.room_labels.index(room) if room in self.room_labels else 0,
            'activity_label': self.activity_labels.index(activity) if activity else 0,
            'station_label': list(self.station_map.keys()).index(station) if station in self.station_map else 0
        }
    
    def _extract_labels_from_filename(self, pcap_file):
        """Extract labels from filename pattern: L_{room}_A_{activity}_*.pcapng"""
        
        filename = pcap_file.stem
        
        # Pattern: L_{room}_A_{activity}_c{config}_n_{num}_AP_{antenna}
        match = re.match(r'L_(\d+)_([A-U])_(\d+)_c(\d+)_n_(\d+)_AP_(.+)', filename)
        if match:
            room_num, activity, _, _, _, _ = match.groups()
            
            # Map room number to room name (you may need to adjust this)
            room_map = {
                '62': 'Classroom',
                '63': 'Kitchen', 
                '64': 'Livingroom'
            }
            room = room_map.get(room_num, 'Unknown')
            
            # Default station and MAC
            station = '9C'  # Default, could be extracted from antenna config
            mac_address = self.station_map.get(station, 'ff:ff:ff:ff:ff:ff')
            
            return {
                'pcap_file': str(pcap_file),
                'mac_address': mac_address,
                'room': room,
                'station': station,
                'activity': activity,
                'room_label': self.room_labels.index(room) if room in self.room_labels else 0,
                'activity_label': self.activity_labels.index(activity) if activity else 0,
                'station_label': list(self.station_map.keys()).index(station) if station in self.station_map else 0
            }
        
        return None
    
    def _load_csi_data(self, pcap_file, mac_address):
        """Load CSI data from PCAP file."""
        
        # Check cache first
        if self.csi_cache is not None and pcap_file in self.csi_cache:
            return self.csi_cache[pcap_file]
        
        # Extract CSI using backend
        ts, vs = csi_backend.get_v_matrix(pcap_file, mac_address, verbose=False)
        
        # Cache if enabled
        if self.csi_cache is not None:
            self.csi_cache[pcap_file] = (ts, vs)
        
        return ts, vs
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        # Load CSI data
        ts, vs = self._load_csi_data(
            sample_info['pcap_file'],
            sample_info['mac_address']
        )
        
        # Handle case where PCAP has no data
        if len(vs) == 0:
            # Return dummy data
            v_tensor = torch.zeros(self.seq_length, 64, 3, 1, dtype=torch.complex64)
        else:
            # Take a random window or the entire sequence
            if len(vs) >= self.seq_length:
                # Random start position for data augmentation during training
                max_start = len(vs) - self.seq_length
                start_idx = np.random.randint(0, max_start + 1) if max_start > 0 else 0
                v_window = vs[start_idx:start_idx + self.seq_length]
            else:
                # Pad if too short
                v_window = np.pad(
                    vs,
                    ((0, self.seq_length - len(vs)), (0, 0), (0, 0), (0, 0)),
                    mode='constant'
                )
            
            v_tensor = torch.tensor(v_window, dtype=torch.complex64)
        
        # Get label based on label_type
        if self.label_type == 'activity':
            label = sample_info['activity_label']
        elif self.label_type == 'room':
            label = sample_info['room_label']
        elif self.label_type == 'station':
            label = sample_info['station_label']
        else:
            label = sample_info['activity_label']  # Default
        
        # Return in format compatible with training system
        # (data, label, user_id, orientation, rx_id)
        return (
            v_tensor,
            torch.tensor(label, dtype=torch.long),
            torch.tensor(sample_info['station_label'], dtype=torch.long),
            torch.tensor(0, dtype=torch.long),  # orientation placeholder
            torch.tensor(0, dtype=torch.long)   # rx_id placeholder
        )
