# Wi-Fi Sensing Library

A unified toolkit for Wi-Fi Sensing, integrating robust BFR extraction (`WiPiCap`) and advanced sensing algorithms (`RF_CRATE`).

## Features
- **Robust Extraction**: Extract Beamforming Reports (V-Matrices) from 802.11ac/ax (VHT/HE) traffic (PCAP files).
- **SU-MIMO & Batching**: Support for Single-User MIMO handling and sliding-window batching for time-series models.
- **Unified Interface**: Single CLI for extraction, training, and inference.

## Installation

```bash
git clone https://github.com/Nirtzur0/wifi-sensing.git
cd wifi-sensing
pip install -e .
```

## Usage

The library provides a unified command-line interface `wifi-sensing`.

### 1. Data Extraction
Extract V-Matrices from a structured raw dataset (folders of PCAP files).
```bash
wifi-sensing extract --data_dir /path/to/dataset_root
```

### 2. Training
Train or fine-tune models (e.g., Widar3, RF_CRATE).
```bash
wifi-sensing train --config_file wifi_sensing_lib/configs/widar3G6/widar3G6_widar3.yaml
```

### 3. Inference
Run inference on a single PCAP file.
```bash
wifi-sensing infer --pcap data.pcap --config path/to/config.yaml --address STA_MAC_ADDRESS
```

## Configuration
- **Model Configs**: Located in `wifi_sensing_lib/configs/`.
- **Training Batching**: Set `seq_length` and `stride` in the YAML config to control sliding window size.

## Project Structure
- `wifi_sensing_lib/`: Main package.
- `backend/`: Low-level processing and CSI extraction (`csi_backend`).
- `data/`: Datasets and data loading utilities.
- `models/`: Model architectures (RF_CRATE, etc.).
- `training/`: Training loops and utilities.
- `inference/`: Inference pipelines.
- `visualization/`: Plotting tools (`BFRPlotter`).
- `cli.py`: Entry point for the CLI.
