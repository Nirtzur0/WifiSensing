import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Unified Wi-Fi Sensing Toolkit")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # --- Extract Command ---
    extract_parser = subparsers.add_parser("extract", help="Extract V-Matrices from PCAP database")
    extract_parser.add_argument("--data_dir", required=True, help="Path to the dataset root directory (containing 'Raw')")
    
    # --- Train Command ---
    train_parser = subparsers.add_parser("train", help="Train or Test RF_CRATE model")
    train_parser.add_argument("--config_file", type=str, required=True, help="Configuration YAML file path")
    train_parser.add_argument("--cuda_index", type=int, default=0, help="The index of the cuda device")
    train_parser.add_argument("--mode", type=int, default=0, help="0: train + test, 1: test only, 2: finetune + test, 3: check the pipeline")
    train_parser.add_argument("--pretrained_model", type=str, default=None, help="The file path of the pretrained model weights")
    
    # --- Infer Command ---
    infer_parser = subparsers.add_parser("infer", help="Run inference on a single PCAP")
    infer_parser.add_argument("--pcap", required=True, help="Path to PCAP file")
    infer_parser.add_argument("--config", required=True, help="Path to YAML config")
    infer_parser.add_argument("--address", default="ff:ff:ff:ff:ff:ff", help="Station MAC address filter")

    # --- Plot Command ---
    plot_parser = subparsers.add_parser("plot", help="Visualize BFR data")
    plot_parser.add_argument("--file", required=True, help="Path to input file (.pt or .pcap)")
    plot_parser.add_argument("--type", choices=['heatmap', 'time_series'], default='heatmap', help="Type of plot")
    plot_parser.add_argument("--mode", choices=['amplitude', 'phase'], default='amplitude', help="Data mode")
    plot_parser.add_argument("--link", default="0,0", help="Link index 'rx,tx' e.g. '0,0'")
    plot_parser.add_argument("--address", default="ff:ff:ff:ff:ff:ff", help="Station MAC (if using PCAP)")

    args = parser.parse_args()
    
    if args.command == "extract":
        from .backend.database import DatabaseExtractor
        print(f"Starting Extraction...")
        extractor = DatabaseExtractor(args.data_dir)
        extractor.process_database()
        
    elif args.command == "train":
        from .training.trainer import run_experiment
        print(f"Starting Training/Testing...")
        run_experiment(args.config_file, args.mode, args.cuda_index, args.pretrained_model)
        
    elif args.command == "infer":
        from .inference.pipeline import InferencePipeline
        print(f"Starting Inference...")
        pipeline = InferencePipeline(args.config)
        pipeline.run(args.pcap, station_address=args.address)

    elif args.command == "plot":
        from .visualization import BFRPlotter
        import torch
        from .backend import csi_backend
        
        file_path = args.file
        v_matrix = None
        
        if file_path.endswith(".pt"):
            print(f"Loading {file_path}...")
            data = torch.load(file_path)
            if isinstance(data, dict) and "v_matrix" in data:
                v_matrix = data["v_matrix"]
            else:
                # Assume raw tensor
                v_matrix = data
        elif file_path.endswith(".pcap") or file_path.endswith(".pcapng"):
            print(f"Extracting from {file_path} (Station: {args.address})...")
            _, vs = csi_backend.get_v_matrix(file_path, args.address, verbose=True)
            if len(vs) == 0:
                print("No data extracted.")
                return
            v_matrix = torch.tensor(vs, dtype=torch.complex64)
        else:
            print("Unknown file format. Use .pt (extracted) or .pcap")
            return

        rx, tx = map(int, args.link.split(','))
        
        if args.type == "heatmap":
            BFRPlotter.plot_heatmap(v_matrix, link=(rx, tx), mode=args.mode)
        elif args.type == "time_series":
            BFRPlotter.plot_time_series(v_matrix, link=(rx, tx), mode=args.mode)
        
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
