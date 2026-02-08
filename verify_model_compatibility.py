#!/usr/bin/env python3
import sys
import torch
import numpy as np
import pandas as pd
from wifi_sensing_lib.data import PcapTrainingDataset
from wifi_sensing_lib.data.interfaces.pcap_dataset import FeatureReshaper
from wifi_sensing_lib.models import registered_models, get_registered_models

def verify_models():
    print("=" * 80)
    print("Verifying Model Compatibility with BFR Data")
    print("=" * 80)

    # 1. Load Data
    data_dir = 'wifi_sensing_lib/data/local_pcaps'
    print(f"Loading data from: {data_dir}")
    
    # Load with basic config just to get raw data
    base_config = {
        'label_type': 'activity',
        'seq_length': 100,  # Standard length
        'stride': 50,
        'station_filter': None,
        # Keep compatibility checks fast: decode a bounded number of BFR frames from the PCAP.
        # The sample PCAP has many VHT BFR packets; decoding the full file via tshark can be slow.
        'num_to_process': 300,
        'verbose': False,
    }
    
    try:
        dataset = PcapTrainingDataset(data_dir, config=base_config)
        if len(dataset) == 0:
            print("ERROR: No samples found in dataset. Ensure local_pcaps has valide files.")
            return
        
        # Get one sample
        raw_data, label, _, _, _ = dataset[0]
        # raw_data is [T, S, Nr, Nc]
        
        # Create a batch of size 2
        batch_raw = raw_data.unsqueeze(0).repeat(2, 1, 1, 1, 1) # [B, T, S, Nr, Nc]
        print(f"Loaded sample batch shape: {batch_raw.shape}")
            
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return

    results = []

    # 2. Test Each Model
    print("\nStarting Model Verification Loop...")
    
    for model_name in registered_models.keys():
        print(f"\n>> Testing model: {model_name}")
        status = "FAILED"
        error_msg = ""
        
        try:
            # Determine Configuration based on Model Family
            model_config = base_config.copy()
            model_config['num_classes'] = 6 # Arbitrary
            model_config['batch_size'] = 2
            
            # Defaults
            target_format = 'complex'
            reshaper_config = {}
            batch_raw_to_use = batch_raw
             
            # Specific Configurations
            if 'swin' in model_name or 'vit' in model_name:
                target_format = 'amplitude' 
                reshaper_config = {'format': 'amplitude', 'model_input_shape': 'BCHW'}
                # Dummy image_size to ensure update block triggers
                model_config['image_size'] = (224, 224) 
                model_config['in_channels'] = 1 
                
            elif 'crate' in model_name and 'rf' not in model_name:
                target_format = 'amplitude'
                reshaper_config = {'format': 'amplitude', 'model_input_shape': 'BCHW'}
                model_config['image_size'] = (224, 224) # Will be updated
                model_config['patch_size'] = 16
                model_config['in_channels'] = 1 
                
            elif 'stfnet' in model_name:
                # STFNet expects [B, T, C] (time-major sequence with flattened features).
                target_format = 'amplitude'
                reshaper_config = {
                    'format': 'amplitude', 
                    'model_input_shape': 'BLC',
                }
                # We'll set dimensions after reshape.
                
            elif 'rf_net' in model_name:
                 target_format = 'amplitude' 
                 reshaper_config = {'format': 'amplitude', 'model_input_shape': 'BCHW'}
                 # Will set dims after reshape
                 
            elif 'slnet' in model_name:
                target_format = 'dfs'
                reshaper_config = {
                    'format': 'dfs',
                    'model_input_shape': 'B2CNFT',
                    'window_size': 64,
                    'window_step': 8
                }
                # Pre-calculate to avoid init failure if possible, but better to update after reshape
                # But get_registered_models is called after this block.
                # Just placeholder configs here.
                model_config['freq_bins'] = 65 
                model_config['time_steps'] = 6 

            elif 'widar3' in model_name:
                target_format = 'dfs'
                reshaper_config = {
                    'format': 'dfs',
                    'model_input_shape': 'BTCHW', 
                    'window_size': 64, 
                    'window_step': 8
                } 
                model_config['time_step'] = 20
                model_config['in_channels'] = 1 
                model_config['hight'] = 20
                model_config['width'] = 20
                
            elif 'rf_crate' in model_name:
                 target_format = 'complex'
                 reshaper_config = {'format': 'complex', 'model_input_shape': 'BCHW'} 
                 model_config['image_size'] = (100, 64) 
                 model_config['patch_size'] = (10, 8)
                 model_config['in_channels'] = 6 
                 model_config['feedforward'] = 128
                 model_config['relu_type'] = 'gelu'
                 model_config['patch_embedding_method'] = 'linear_patch' 
                 model_config['mlp_head_type'] = 'linear'
                 model_config['output_type'] = 'cls'


            # 2a. Reshape Data
            print(f"   Shape Config: {target_format}, {reshaper_config}")
            reshaper = FeatureReshaper(reshaper_config)
            
            # Transform
            try:
                # Slice input to avoid OOM for large models/complex data
                batch_raw_to_use = batch_raw[:, :, :, :1, :4] if 'slnet' in model_name or 'stfnet' in model_name or 'rf_net' in model_name or 'widar3' in model_name else batch_raw

                model_input = reshaper.shape_convert(batch_raw_to_use)
                print(f"   Reshaped Input: {model_input.shape}")
                
                # STFNet: ensure the feature dimension is compatible with the FFT-branch split.
                # The current implementation assumes C is divisible by 8.
                if 'stfnet' in model_name and len(model_input.shape) == 3:
                     b, t, c = model_input.shape
                     c8 = (c // 8) * 8
                     if c8 != c:
                         model_input = model_input[:, :, :c8]
                         print(f"   Truncated STFNet feature dim from {c} to {c8}: {model_input.shape}")

                # RFNet Reshape Logic: [B, C, H, W] -> [B, H, C*W]
                if 'rf_net' in model_name and len(model_input.shape) == 4:
                     b, c, t, f = model_input.shape
                     model_input = model_input.permute(0, 2, 1, 3).reshape(b, t, c*f)
                     print(f"   Adjusted RFNet Input to [B, T, C]: {model_input.shape}")

                # Widar3 Resize
                if 'widar3' in model_name and len(model_input.shape) == 5:
                     b, t, c, h, w = model_input.shape
                     flat = model_input.contiguous().view(-1, 1, h, w)
                     resized = torch.nn.functional.interpolate(flat, size=(20, 20), mode='bilinear')
                     model_input = resized.view(b, t, c, 20, 20)
                     print(f"   Resized Widar3 Input: {model_input.shape}")

                # FIX: Truncate width for Crate/RF Crate if not divisible by patch size
                if 'patch_size' in model_config:
                    if isinstance(model_config['patch_size'], tuple):
                        ph, pw = model_config['patch_size']
                    else:
                        ph = pw = model_config['patch_size']
                    
                    if len(model_input.shape) == 4: # B, C, H, W
                         H, W = model_input.shape[2], model_input.shape[3]
                         new_H = (H // ph) * ph
                         new_W = (W // pw) * pw
                         if new_H != H or new_W != W:
                             print(f"   Truncating input from {H}x{W} to {new_H}x{new_W} for patch divisibility")
                             model_input = model_input[:, :, :new_H, :new_W]
            
            except Exception as e:
                print(f"   RESHAPE ERROR: {e}")
                import traceback
                traceback.print_exc()
                status = "RESHAPE_FAIL"
                error_msg = str(e)
                results.append({'Model': model_name, 'Status': status, 'Error': error_msg})
                continue

            # Update Model Config based on actual input shape
            if isinstance(model_input, torch.Tensor):
                 if 'stfnet' in model_name:
                      # [B, T, C]
                      model_config['time_length'] = model_input.shape[1]
                      model_config['act_domain'] = 'freq' # Fix KeyError
                      # STFNet expects 'in_channels' to be Total Channels for validation?
                      # In init: sensor_in_channels = config['in_channels'] // sensor_num
                      # We flattened C*F -> C_total.
                      # Let's set sensor_num=1, in_channels=C_total.
                      model_config['sensor_num'] = 1
                      model_config['feature_dim'] = model_input.shape[2]
                      model_config['in_channels'] = model_input.shape[2]

                 elif 'rf_net' in model_name:
                      # [B, T, C]
                      model_config['time_length'] = model_input.shape[1]
                      model_config['in_channels'] = model_input.shape[2]

                 elif 'slnet' in model_name:
                      # [B, 2, S, Ant, F, T]
                      # S = dim 2, Ant = dim 3
                      S = model_input.shape[2]
                      Ant = model_input.shape[3]
                      model_config['sensor_num'] = Ant
                      model_config['in_channels'] = S * Ant # Total channels
                      model_config['time_steps'] = model_input.shape[5]
                 
                 elif 'widar3' in model_name:
                      # [B, T, C, H, W]
                      model_config['in_channels'] = model_input.shape[2]
                      model_config['time_step'] = model_input.shape[1] # Update time_step from actual input

                 elif len(model_input.shape) == 4:
                      # [B, C, H, W] Standard
                      model_config['in_channels'] = model_input.shape[1]
                      model_config['image_size'] = (model_input.shape[2], model_input.shape[3])
                           
            # 2b. Initialize Model
            try:
                model_ret = get_registered_models(model_name, model_config)
                if model_ret is None:
                    print("   Init returned None")
                    status = "INIT_FAIL"
                    error_msg = "get_registered_models returned None"
                    results.append({'Model': model_name, 'Status': status, 'Error': error_msg})
                    continue
                
                # Handle tuple return (model, decoder) for recon models
                if isinstance(model_ret, tuple):
                    model = model_ret[0]
                    # We can optionally test decoder too, but for now just model compatibility
                else:
                    model = model_ret
                    
            except Exception as e:
                print(f"   INIT ERROR: {e}")
                import traceback
                traceback.print_exc()
                status = "INIT_FAIL"
                error_msg = str(e)
                results.append({'Model': model_name, 'Status': status, 'Error': error_msg})
                continue
                
            # 2c. Forward Pass
            try:
                # STFNet specific adjustment
                # Removed previous permute logic as we handle it in Reshape block now
                
                output = model(model_input)
                
                # Handle tuple return (e.g. RFNet, RF-Crate Recon)
                if isinstance(output, tuple):
                    output = output[0]
                    
                print(f"   Output Shape: {output.shape}")
                status = "PASS"
            except Exception as e:
                print(f"   FORWARD ERROR: {e}")
                import traceback
                traceback.print_exc()
                status = "FORWARD_FAIL"
                error_msg = str(e)
                
        except Exception as e:
            print(f"   UNEXPECTED ERROR: {e}")
            status = "CRASH"
            error_msg = str(e)
            
        results.append({'Model': model_name, 'Status': status, 'Error': error_msg})

    # 3. Report
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    df = pd.DataFrame(results)
    print(df.to_string())
    
    # Save to file
    df.to_csv('compatibility_report.csv', index=False)
    print("\nReport saved to compatibility_report.csv")

if __name__ == '__main__':
    verify_models()
