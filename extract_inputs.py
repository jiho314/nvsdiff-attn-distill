import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from easydict import EasyDict

from src import datasets

def parse_args():
    # ... (argument parsing is the same) ...
    parser = argparse.ArgumentParser(description="Extract the first input view from each scene in a dataset.")
    parser.add_argument("--test_dataset_config", type=str, required=True, help="Path to the dataset configuration file (e.g., configs/datasets/test.yaml).")
    parser.add_argument("--test_dataset", type=str, required=True, help="The name of the dataset to use from the config file (e.g., re10k_wds_test).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where the extracted images will be saved.")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes to process. Set to -1 to process all scenes.")
    return parser.parse_args()

def main():
    args = parse_args()

    # ... (Setup and Dataset Loading are the same) ...
    print(f"Creating output directory at: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    print("Loading dataset configuration...")
    test_dataset_config = EasyDict(OmegaConf.load(args.test_dataset_config))[args.test_dataset]
    print(f"Initializing dataset: '{args.test_dataset}'")
    test_dataset = datasets.__dict__[test_dataset_config.cls_name](**test_dataset_config.config)
    dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )
    print("Dataset loaded successfully.")

    num_to_process = len(dataloader) if args.num_scenes == -1 else args.num_scenes
    print(f"Starting extraction of {num_to_process} scenes...")

    # --- 3. Loop, Extract, and Save (with updated logic) ---
    for i, batch in enumerate(tqdm(dataloader, total=num_to_process)):
        if args.num_scenes != -1 and i >= args.num_scenes:
            print(f"Reached specified limit of {args.num_scenes} scenes. Stopping.")
            break

        try:
            # ✅ **Step 1: Get the real identifier from the batch.**
            # The ID is likely a string inside a list, so we take the first element.
            original_id = batch['tag'][0]
            
            # ✅ **Step 2: Sanitize the ID for use in a filename.**
            # This replaces characters like '/' with '_' to prevent path issues.
            sanitized_id = original_id.replace('/', '_').replace('\\', '_')

            image_tensor_all_frames = batch['image'][0]
            first_input_view_tensor = image_tensor_all_frames[0]

            # ✅ **Step 3: Use the sanitized ID in the filename.**
            filename = f"{sanitized_id}_input_1.jpg"
            output_path = os.path.join(args.output_dir, filename)

            # ... (image conversion and saving is the same) ...
            img_np = (first_input_view_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(img_np).save(output_path)

        except Exception as e:
            print(f"Could not process scene #{i}. Error: {e}")
            continue
    
    print("\nExtraction complete!")

if __name__ == "__main__":
    main()