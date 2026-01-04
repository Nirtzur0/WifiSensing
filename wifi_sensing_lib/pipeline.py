import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from .wipicap_adapter import WifiSensingDataset, FeatureReshaper
from .rfcrate_adapter import load_model

class InferencePipeline:
    """
    Orchestrates the Wi-Fi sensing inference process:
    1. Loads configuration.
    2. Initializes data loading (from PCAP).
    3. Loads the model.
    4. Runs inference.
    """
    def __init__(self, config_path, device=None):
        self.config = self._load_config(config_path)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        self.converter = FeatureReshaper(self.config)

    def _load_config(self, path):
        with open(path, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def _load_model(self):
        model_name = self.config['model_name']
        try:
            model = load_model(model_name, self.config, self.device)
            model.eval()
            print(f"Model '{model_name}' loaded successfully on {self.device}.")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_name}': {e}")

    def run(self, pcap_file, station_address="ff:ff:ff:ff:ff:ff"):
        """
        Runs inference on the provided PCAP file for the specific station address.
        """
        print(f"Initializing dataset for {pcap_file} (Station: {station_address})...")
        dataset = WifiSensingDataset(pcap_file, address=station_address, config=self.config)
        
        if len(dataset) == 0:
            print("No samples found in dataset.")
            return

        loader = DataLoader(dataset, batch_size=self.config.get('batch_size', 1), shuffle=False)
        
        print("Starting Inference...")
        with torch.no_grad():
            for i, (v, label, _, _, _) in enumerate(loader):
                v = v.to(self.device)
                
                # Reshape/Format features
                x = self.converter.shape_convert(v)
                print(f"Batch {i}: Input shape {x.shape}")
                
                # output = self.model(x)
                # print(f"Batch {i}: Output shape {output.shape}")
                
                # For verification, we stop after first batch
                if i >= 0:
                    break 

def main():
    parser = argparse.ArgumentParser(description="Wi-Fi Sensing Inference Pipeline")
    parser.add_argument("--pcap", required=True, help="Path to PCAP file")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--address", default="ff:ff:ff:ff:ff:ff", help="Station MAC address filter")
    args = parser.parse_args()
    
    pipeline = InferencePipeline(args.config)
    pipeline.run(args.pcap, station_address=args.address)

if __name__ == "__main__":
    main()
