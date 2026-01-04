# Wi-Fi Sensing Library

A unified library for Wi-Fi sensing, integrating `WiPiCap` for BFR extraction and `RF_CRATE` for deep learning models.

## Features

- **WiPiCap Integration**: Extracts Beamforming Reports (V-Matrices) from `.pcap` files using efficient Cython extensions.
- **RF_CRATE Models**: Provides a suite of PyTorch models (e.g., Widar3, STFNet, RF-Net) for Wi-Fi sensing tasks.
- **Unified Pipeline**: A simple API to process PCAP files and run inference.
- **Database Extractor**: Tools to batch process raw dataset directories.

## Installation

```bash
pip install -e .
```

## Usage

### Extractor
Process a raw dataset directory structure:
```bash
python3 -m wifi_sensing_lib.extractor --data_dir /path/to/Data/DatasetName
```

### Inference Pipeline
Run models on a capture file:
```bash
python3 -m wifi_sensing_lib.pipeline --pcap /path/to/capture.pcap --config wifi_sensing_lib/rfcrate/Configurations/widar3G6/widar3G6_widar3.yaml
```
