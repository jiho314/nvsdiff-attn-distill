import sys
from pathlib import Path
from PIL import Image

# --- Configuration ---

# Set the root directory to search from.
# Path(".") means the directory where you run the script.
ROOT_DIRECTORY = Path("./vis_attn_re10k/2025-10-22_17-28-06/naive_100k_qual/per_view/700/sample_4")

# The name of the grid image file to find.
TARGET_FILENAME = "VIS_STACKED.png"

# The size (width and height) of each grid cell.
GRID_SIZE = 512

# --- End of Configuration ---

def process_image(image_path: Path):
    """
    Reads a 2x3 grid image and saves its 5 unique components.

    Grid layout is assumed to be (each cell is 512x512):
    [reference, target1, target2]
    [reference, pred1,   pred2  ]
    """
    print(f"Processing {image_path}...")
    parent_dir = image_path.parent

    try:
        # Open the image using a context manager
        with Image.open(image_path) as img:
            
            # Verify image dimensions based on the 2x3 grid of 512px tiles
            # Width = 3 * 512 = 1536
            # Height = 2 * 512 = 1024
            expected_width = 3 * GRID_SIZE
            expected_height = 2 * GRID_SIZE

            if img.size != (expected_width, expected_height):
                print(f"  [WARN] Skipping {image_path}:")
                print(f"    Expected dimensions ({expected_width}, {expected_height}), but got {img.size}.")
                return

            S = GRID_SIZE # Alias for brevity

            # Define the crops: (filename, (left, upper, right, lower))
            # Note: Image.crop() box is [left, upper, right, lower]
            crops_to_save = [
                # Row 1
                ("reference.png", (0*S, 0*S, 1*S, 1*S)),
                ("target1.png",   (1*S, 0*S, 2*S, 1*S)),
                ("target2.png",   (2*S, 0*S, 3*S, 1*S)),
                
                # Row 2
                # Skip (0*S, 1*S, 1*S, 2*S) as it's a duplicate of 'reference'
                ("pred1.png",     (1*S, 1*S, 2*S, 2*S)),
                ("pred2.png",     (2*S, 1*S, 3*S, 2*S)),
            ]

            # Crop and save each part
            for filename, box in crops_to_save:
                # Crop operation is lazy, so use a context manager
                with img.crop(box) as cropped_img:
                    output_path = parent_dir / filename
                    # Save the cropped image
                    cropped_img.save(output_path)
                    try:
                        # Try to print a relative path for cleaner logs
                        print_path = output_path.relative_to(ROOT_DIRECTORY)
                    except ValueError:
                        # Fallback if not in ROOT_DIRECTORY (shouldn't happen)
                        print_path = output_path
                    print(f"  Saved {print_path}")

    except Exception as e:
        print(f"  [ERROR] Failed to process {image_path}: {e}")

def main():
    """
    Finds all TARGET_FILENAME files recursively from ROOT_DIRECTORY and processes them.
    """
    
    # Resolve to an absolute path for clear logging
    abs_root = ROOT_DIRECTORY.resolve()
    print(f"Starting grid image processing in: {abs_root}")
    print(f"Looking for files named: {TARGET_FILENAME}\n")

    # Use glob to find all matching files recursively
    try:
        # Using list() to find all files first, which can be memory-intensive
        # for *massive* directories, but is simpler to log.
        image_paths = list(ROOT_DIRECTORY.glob(f"**/{TARGET_FILENAME}"))
    except Exception as e:
        print(f"Error while searching for files: {e}", file=sys.stderr)
        return

    if not image_paths:
        print(f"No '{TARGET_FILENAME}' files found in {abs_root}.")
        print("Please check ROOT_DIRECTORY in the script or run it from the correct folder.")
        return

    print(f"Found {len(image_paths)} file(s) to process.\n")

    for path in image_paths:
        process_image(path)

    print("\nProcessing complete.")

if __name__ == "__main__":
    # Before running, ensure you have Pillow installed:
    # pip install Pillow
    main()
