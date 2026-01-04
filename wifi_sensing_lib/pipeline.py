import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from .wipicap_adapter import WiPiCapDataset, WiPiCapDataShapeConverter
from .rfcrate_adapter import load_model

def run_pipeline(pcap_file, config_file):
    # Load config
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Init Dataset
    dataset = WiPiCapDataset(pcap_file, address="ff:ff:ff:ff:ff:ff", config=config) # Address needs to be correct!
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Init Model
    model_name = config['model_name']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = load_model(model_name, config, device)
        model.eval()
        print(f"Model {model_name} loaded.")
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        return

    # Init Converter
    converter = WiPiCapDataShapeConverter(config)
    
    # Inference Loop
    print("Starting Inference...")
    with torch.no_grad():
        for i, (v, label, _, _, _) in enumerate(loader):
            v = v.to(device)
            # v shape: [Batch, Subc, Nr, Nc]
            
            x = converter.shape_convert(v)
            print(f"Batch {i}: Input shape {x.shape}")
            
            # output = model(x)
            # print(f"Batch {i}: Output shape {output.shape}")
            
            if i >= 0: # Just one batch for verification
                break 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcap", required=True, help="Path to PCAP file")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    
    run_pipeline(args.pcap, args.config)
