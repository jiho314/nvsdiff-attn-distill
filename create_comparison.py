import os
import shutil
from PIL import Image, ImageDraw, ImageFont

# --- Configuration ---

# 1. Define the two source directories
DIR_NAIVE = "/mnt/data1/jiho/vggt-nvs/nvsdiff-attn-distill/vis_full_perview/2025-10-27_16-12-52/full_40k/full/250"
DIR_OURS = "/mnt/data1/jiho/vggt-nvs/nvsdiff-attn-distill/vis_full_perview/2025-10-27_16-50-07/full_40k/full/250"

# 2. Define the list of indexes to process
INDEXES = [
    2, 6, 8, 9, 10, 11, 14, 15, 21, 22, 23, 26, 28, 41, 42, 46, 52, 66, 90, 94, 98
]

# 3. Define the main output directory
OUTPUT_DIR = "output_comparison"

# 4. Define styling for the stitched image
LABEL_SPACE = 50       # Height of the top padding for labels
DIVIDER_WIDTH = 5      # Width of the line between images
BG_COLOR = (255, 255, 255) # White
TEXT_COLOR = (0, 0, 0)       # Black
DIVIDER_COLOR = (0, 0, 0)    # Black
FONT_SIZE = 30
IMAGE_FILENAME = "VIS_STACKED.png"

# --- End of Configuration ---


def load_font(size):
    """Tries to load a common font, falls back to default if not found."""
    try:
        # Try loading a common font
        font = ImageFont.truetype("DejaVuSans.ttf", size=size)
    except IOError:
        try:
            # Try another common font (Windows)
            font = ImageFont.truetype("arial.ttf", size=size)
        except IOError:
            # Fallback to default bitmap font
            print("Warning: Could not load 'DejaVuSans.ttf' or 'arial.ttf'. Falling back to default font.")
            font = ImageFont.load_default()
    return font

def process_images():
    print(f"Starting comparison process. Output will be in: {OUTPUT_DIR}\n")
    
    # Ensure the main output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load the font
    font = load_font(FONT_SIZE)
    
    processed_count = 0
    error_count = 0

    for index in INDEXES:
        sample_name = f"sample_{index}"
        print(f"--- Processing {sample_name} ---")
        
        try:
            # 1. Define all paths
            src_naive_path = os.path.join(DIR_NAIVE, sample_name, IMAGE_FILENAME)
            src_ours_path = os.path.join(DIR_OURS, sample_name, IMAGE_FILENAME)
            
            # Create a dedicated output folder for this sample
            sample_output_dir = os.path.join(OUTPUT_DIR, sample_name)
            os.makedirs(sample_output_dir, exist_ok=True)
            
            dst_naive_path = os.path.join(sample_output_dir, f"naive_{IMAGE_FILENAME}")
            dst_ours_path = os.path.join(sample_output_dir, f"ours_{IMAGE_FILENAME}")
            dst_stitched_path = os.path.join(sample_output_dir, "comparison_stitched.png")

            # 2. Task 1: Copy original files
            print(f"Copying original files for {sample_name}...")
            shutil.copy2(src_naive_path, dst_naive_path)
            shutil.copy2(src_ours_path, dst_ours_path)
            
            # 3. Task 2: Create stitched file
            print(f"Creating stitched image for {sample_name}...")
            
            # Open images
            with Image.open(src_naive_path) as img_naive, \
                 Image.open(src_ours_path) as img_ours:
                
                # Assume images are the same size
                if img_naive.size != img_ours.size:
                    print(f"Warning: Images for {sample_name} are different sizes. Skipping stitching.")
                    error_count += 1
                    continue
                
                width, height = img_naive.size
                
                # Calculate dimensions for the new canvas
                new_width = width + DIVIDER_WIDTH + width
                new_height = LABEL_SPACE + height
                
                # Create the new canvas
                stitched_img = Image.new("RGB", (new_width, new_height), BG_COLOR)
                draw = ImageDraw.Draw(stitched_img)
                
                # Paste the images
                stitched_img.paste(img_naive, (0, LABEL_SPACE))
                stitched_img.paste(img_ours, (width + DIVIDER_WIDTH, LABEL_SPACE))
                
                # Draw the divider line
                draw.line(
                    [(width, LABEL_SPACE), (width, new_height)],
                    fill=DIVIDER_COLOR,
                    width=DIVIDER_WIDTH
                )
                
                # Draw the labels
                # Get text bounding box to center it
                naive_bbox = draw.textbbox((0, 0), "Naive", font=font)
                naive_text_width = naive_bbox[2] - naive_bbox[0]
                naive_text_height = naive_bbox[3] - naive_bbox[1]
                
                ours_bbox = draw.textbbox((0, 0), "Ours", font=font)
                ours_text_width = ours_bbox[2] - ours_bbox[0]

                # Calculate positions
                naive_text_x = (width - naive_text_width) / 2
                ours_text_x = (width + DIVIDER_WIDTH) + (width - ours_text_width) / 2
                text_y = (LABEL_SPACE - naive_text_height) / 2 # Center vertically in the label space
                
                draw.text((naive_text_x, text_y), "Naive", fill=TEXT_COLOR, font=font)
                draw.text((ours_text_x, text_y), "Ours", fill=TEXT_COLOR, font=font)
                
                # Save the final stitched image
                stitched_img.save(dst_stitched_path)
                print(f"Successfully saved {dst_stitched_path}")
                processed_count += 1

        except FileNotFoundError as e:
            print(f"Error: Could not find file for {sample_name}. Skipping.")
            print(f"Missing file: {e.filename}")
            error_count += 1
        except Exception as e:
            print(f"An unexpected error occurred for {sample_name}: {e}")
            error_count += 1
        
        print("--- Done --- \n")

    print("==========================================")
    print("Comparison process finished.")
    print(f"Successfully processed: {processed_count} samples")
    print(f"Failed or skipped:    {error_count} samples")
    print("==========================================")


if __name__ == "__main__":
    process_images()