# PCAP Training Dataset - Usage Guide

## Overview

The PCAP training dataset system automatically extracts labels from directory structure and filenames, enabling training on PCAP files without manual labeling.

## Directory Structure

The system supports two directory structures:

### 1. Processed Directory Structure (Recommended)
```
Data/
└── MyDataset/
    └── Processed/
        ├── Classroom/
        │   ├── 9C/
        │   │   └── FeedBack_Pcap/
        │   │       ├── L_62_A_12_c1_n_1_AP_4x4-2_9C.pcapng
        │   │       └── ...
        │   ├── 25/
        │   └── 89/
        ├── Kitchen/
        └── Livingroom/
```

Labels extracted:
- **Room**: Classroom, Kitchen, Livingroom
- **Station**: 9C, 25, 89 (device MAC addresses)
- **Activity**: Extracted from filename (A-U)

### 2. Flat Structure with Labeled Filenames
```
Data/
└── MyDataset/
    ├── L_62_A_12_c1_n_1_AP_4x4-2.pcapng
    ├── L_63_B_15_c2_n_2_AP_4x4-2.pcapng
    └── ...
```

Filename pattern: `L_{room}_A_{activity}_c{config}_n_{num}_AP_{antenna}.pcapng`

## Quick Start

### 1. Organize Your PCAP Files

Use the bash script or manually organize files:
```bash
# Using the provided script
./pcap_logic MyDataset

# Or manually create structure
mkdir -p Data/MyDataset/Processed/Classroom/9C/FeedBack_Pcap
```

### 2. Create Configuration File

Copy and modify `wifi_sensing_lib/configs/pcap_example.yaml`:

```yaml
dataset_name: 'pcap'
dataset_path: 'Data/MyDataset'
label_type: 'activity'  # or 'room', 'station'
seq_length: 100
num_classes: 21  # 21 activities (A-U)
```

### 3. Train the Model

```bash
# Using the training script
python -m wifi_sensing_lib.training.trainer --config_file wifi_sensing_lib/configs/pcap_example.yaml

# Or programmatically
from wifi_sensing_lib.training import run_experiment
run_experiment('wifi_sensing_lib/configs/pcap_example.yaml', mode=0)
```

### 4. Run Inference

```python
from wifi_sensing_lib import InferencePipeline

pipeline = InferencePipeline('path/to/config.yaml')
results = pipeline.run('path/to/new_file.pcapng', station_address='B0:B9:8A:63:55:9C')
```

## Label Types

### Activity Classification (Default)
```yaml
label_type: 'activity'
num_classes: 21  # A-U
```

### Room Classification
```yaml
label_type: 'room'
num_classes: 3  # Classroom, Kitchen, Livingroom
```

### Station Classification
```yaml
label_type: 'station'
num_classes: 3  # 9C, 25, 89
```

## Station MAC Address Mapping

```python
'9C': 'B0:B9:8A:63:55:9C'
'25': '38:94:ED:12:3C:25'
'89': 'CC:40:D0:57:EA:89'
```

## Advanced Options

### Filter by Station
```yaml
station_filter: ['9C', '25']  # Only load these stations
```

### Enable CSI Caching
```yaml
cache_csi: true  # Faster training but uses more memory
```

### Custom Data Format
```yaml
format: 'dfs'  # Options: complex, amplitude, polar, cartesian, dfs
model_input_shape: 'B2CNFT'
```

## Programmatic Usage

```python
from wifi_sensing_lib.data import PcapTrainingDataset

config = {
    'label_type': 'activity',
    'seq_length': 100,
    'stride': 50,
    'format': 'complex',
    'cache_csi': False
}

dataset = PcapTrainingDataset(
    data_dir='Data/MyDataset',
    config=config
)

print(f"Loaded {len(dataset)} samples")

# Get a sample
data, label, user_id, orientation, rx_id = dataset[0]
print(f"Data shape: {data.shape}, Label: {label}")
```

## Troubleshooting

### No samples found
- Check directory structure matches expected format
- Verify PCAP files have `.pcapng` extension
- Check that filenames contain activity labels

### Label extraction fails
- Ensure filenames follow pattern: `*_{ACTIVITY}_*.pcapng`
- Activity must be A-U (single uppercase letter)
- Check room names match: Classroom, Kitchen, Livingroom

### Out of memory
- Reduce `batch_size` in config
- Set `cache_csi: false`
- Reduce `seq_length`

## Example Training Session

```bash
# 1. Organize data
./pcap_logic MyDataset

# 2. Train
python -m wifi_sensing_lib.training.trainer \
    --config_file wifi_sensing_lib/configs/pcap_example.yaml \
    --mode 0

# 3. Test
python -m wifi_sensing_lib.training.trainer \
    --config_file wifi_sensing_lib/configs/pcap_example.yaml \
    --mode 1 \
    --pretrained_model checkpoints/model.pth
```
