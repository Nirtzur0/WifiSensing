#!/usr/bin/env python3
import argparse
import yaml
import os
import sys
import torch

# Ensure the current directory is in the path to allow imports
sys.path.append(os.getcwd())

try:
    from wifi_sensing_lib.training.trainer import run_experiment
except ImportError as e:
    print(f"Error importing library: {e}")
    print("Ensure you are running this script from the root of the project.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Simple CLI for Wi-Fi Sensing Pipeline Orchestration")
    
    # Core Arguments
    parser.add_argument("--model", type=str, default="swin_t", 
                        help="Model to use (e.g., swin_t, rf_crate_tiny, stfnet_standard, slnet_standard)")
    parser.add_argument("--data", type=str, default="wifi_sensing_lib/data/local_pcaps",
                        help="Path to directory containing PCAP files")
    parser.add_argument("--task", type=str, choices=["activity", "room", "station"], default="activity",
                        help="Prediction task / Label type")
    
    # Training Parameters
    parser.add_argument("--mode", type=str, choices=["train", "test", "check"], default="check",
                        help="Pipeline mode: 'train' (train+test), 'test' (inference only), 'check' (1 epoch integration test)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (ignored in check mode)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    
    # Advanced
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained .pth model (required for test mode)")
    parser.add_argument("--output_dir", type=str, default="experiments", help="Directory to save logs and checkpoints")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use")

    args = parser.parse_args()

    # Map mode string to integer expected by trainer
    mode_map = {"train": 0, "test": 1, "check": 3}
    mode_int = mode_map[args.mode]

    # Validate data path
    if not os.path.exists(args.data):
        print(f"Error: Data directory '{args.data}' does not exist.")
        sys.exit(1)

    # 1. Load Base Config (Template)
    # We use pcap_example.yaml as a conceptual template or fall back to defaults
    base_config_path = "wifi_sensing_lib/configs/pcap_example.yaml"
    config = {}
    
    if os.path.exists(base_config_path):
        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    else:
        # Fallback defaults
        config = {
             "dataset_name": "pcap",
             "data_split": [0.7, 0.15, 0.15],
             "optimizer": "AdamW",
             "metric": "accuracy",
             "num_workers": 2,
             "init_rand_seed": 42
        }

    # 2. Update Config with CLI Args
    config["dataset_path"] = os.path.abspath(args.data)
    config["model_name"] = args.model
    config["label_type"] = args.task
    config["num_epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["lr"] = args.lr
    config["cuda_index"] = args.gpu

    # Keep "check" mode fast and deterministic on large PCAPs.
    # Users can override by setting `num_to_process` explicitly in their YAML.
    if args.mode == "check" and "num_to_process" not in config:
        config["num_to_process"] = 30

    # Override paths to keep workspace clean
    os.makedirs(args.output_dir, exist_ok=True)
    config["tensorboard_folder"] = os.path.join(args.output_dir, "runs/")
    config["trained_model_folder"] = os.path.join(args.output_dir, "checkpoints/")
    config["log_folder"] = os.path.join(args.output_dir, "logs/")

    # 3. Model-Specific Heuristics (Defaults for ease of use)
    # These override the YAML defaults to ensure the model actually runs with valid inputs
    
    # RF-Crate family usually needs Complex input
    if "crate" in args.model:
        config["format"] = "complex"
        # Complex models expect channel-first (antennas as channels).
        config["model_input_shape"] = "BCHW-C"
        # Ensure patch sizes exist if not present
        if "patch_size" not in config:
            config["patch_size"] = [10, 8]
            config["image_size"] = [100, 64]
            
    # Swin / ViT usually work well with Amplitude
    elif "swin" in args.model or "vit" in args.model:
        config["format"] = "amplitude"
        config["model_input_shape"] = "BCHW"
        # Match data dimensions observed (40 channels, 100x137)
        config["in_channels"] = 40
        config["image_size"] = [100, 137]
        config["window_size"] = [10, 10] if "window_size" not in config else config["window_size"]
        config["patch_size"] = [4, 4] if "patch_size" not in config else config["patch_size"]
        
    # STFNet needs specific handling (Amplitude + BCHW as tested)
    elif "stfnet" in args.model:
         config["format"] = "amplitude"
         config["model_input_shape"] = "BCHW" # Verification showed this works best
         # Ensure required STFNet params
         if "feature_dim" not in config: config["feature_dim"] = 512 # Default
         config["act_domain"] = "freq"
    
    # SLNet / Widar3 usually use DFS
    elif "slnet" in args.model or "widar3" in args.model:
        config["format"] = "dfs"
        config["model_input_shape"] = "BTCHW" # or B2CNFT depending on implementation details
        # Specific configs might be needed, relying on template for detailed params
    
    # default fallback
    if "format" not in config:
        config["format"] = "amplitude"
        config["model_input_shape"] = "BCHW"

    # 3.5 Fix Layout for Trainer compatibility
    # The trainer requires an 'all_dataset' dictionary for in-domain experiments
    if "all_dataset" not in config:
        config["all_dataset"] = {
            "dataset_path": config["dataset_path"],
            "label_type": config["label_type"],
            "format": config["format"],
            # Copy other essentials that might be nested
            "seq_length": config.get("seq_length", 100),
            "stride": config.get("stride", 50)
        }

    # 3.6 (Check Mode) Infer PCAP shape for sensible defaults
    # This makes `--mode check` work out-of-the-box on the bundled example PCAPs.
    if args.mode == "check" and config.get("dataset_name") == "pcap":
        try:
            from wifi_sensing_lib.backend import csi_backend

            # Pick the first PCAP/PCAPNG found under the dataset path.
            import glob
            pcap_candidates = sorted(
                glob.glob(os.path.join(config["dataset_path"], "**", "*.pcap*"), recursive=True)
            )
            if pcap_candidates:
                station_address = config.get("station_address", "B0:B9:8A:63:55:9C")
                ts, vs = csi_backend.get_v_matrix(
                    pcap_candidates[0],
                    station_address,
                    num_to_process=config.get("num_to_process", None),
                    verbose=False,
                )
                if getattr(vs, "size", 0) > 0:
                    subc, nr, nc = vs.shape[1], vs.shape[2], vs.shape[3]
                    config["in_channels"] = int(nr * nc)

                    # Ensure image_size is compatible with RF-CRATE divisibility constraints.
                    seq_len = int(config.get("seq_length", 100))
                    if "patch_size" in config:
                        patch_w = int(config["patch_size"][1])
                        if patch_w > 0:
                            subc = subc - (subc % patch_w)
                            subc = max(subc, patch_w)
                    config["image_size"] = [seq_len, int(subc)]
        except Exception as e:
            print(f"[!] Shape inference skipped/failed: {e}")

    # 4. Save Temporary Config
    temp_config_path = os.path.join(args.output_dir, "current_run_config.yaml")
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    print("\n" + "="*50)
    print(f" WI-FI SENSING PIPELINE ORCHESTRATOR")
    print("="*50)
    print(f" Model:       {args.model}")
    print(f" Task:        {args.task}")
    print(f" Data:        {args.data}")
    print(f" Mode:        {args.mode.upper()}")
    print(f" Device:      {'GPU ' + str(args.gpu) if torch.cuda.is_available() else 'CPU'}")
    print(f" Config:      {temp_config_path}")
    print("="*50 + "\n")

    # 5. Run Experiment
    try:
        run_experiment(temp_config_path, mode=mode_int, cuda_index=args.gpu, pretrained_model=args.pretrained)
    except KeyboardInterrupt:
        print("\n\n[!] Pipeline stopped by user.")
    except Exception as e:
        print(f"\n[X] Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
