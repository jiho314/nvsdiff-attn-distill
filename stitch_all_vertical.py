import os
from PIL import Image, ImageDraw, ImageFont

# --- Configuration ---

# 1. Define the two source directories
DIR_NAIVE = "/mnt/data1/jiho/vggt-nvs/nvsdiff-attn-distill/vis_full_perview/2025-10-27_16-12-52/full_40k/full/250"
DIR_OURS = "/mnt/data1/jiho/vggt-nvs/nvsdiff-attn-distill/vis_full_perview/2025-10-27_16-50-07/full_40k/full/250"

# 2. Define the list of indexes to process
INDEXES = [
    2, 6, 8, 9, 10, 11, 14, 15, 21, 22, 23, 26, 28, 41, 42, 46, 52, 66, 90, 94, 98
]

# 3. Define the *final* output file name
FINAL_OUTPUT_FILE = "final_vertical_comparison.png"

# 4. Define styling
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
        font = ImageFont.truetype("DejaVuSans.ttf", size=size)
    except IOError:
        try:
            font = ImageFont.truetype("arial.ttf", size=size)
        except IOError:
            print("Warning: Could not load 'DejaVuSans.ttf' or 'arial.ttf'. Falling back to default font.")
            font = ImageFont.load_default()
    return font

def create_vertical_stitch():
    print(f"Starting vertical stitch process. Output will be: {FINAL_OUTPUT_FILE}\n")
    
    font = load_font(FONT_SIZE)
    images_to_stack = [] # This will hold the header + all horizontal rows
    
    total_width = 0
    base_img_width = 0
    base_img_height = 0
    processed_count = 0
    error_count = 0

    if not INDEXES:
        print("No indexes provided. Exiting.")
        return

    # --- 1. Get dimensions and create header ---
    try:
        # Load the first image to get dimensions
        first_index = INDEXES[0]
        sample_name = f"sample_{first_index}"
        first_img_path = os.path.join(DIR_NAIVE, sample_name, IMAGE_FILENAME)
        
        with Image.open(first_img_path) as img:
            base_img_width, base_img_height = img.size
        
        total_width = (base_img_width * 2) + DIVIDER_WIDTH
        print(f"Base image size: {base_img_width}x{base_img_height}")
        print(f"Total canvas width: {total_width}")

        # Create the header image
        header_img = Image.new("RGB", (total_width, LABEL_SPACE), BG_COLOR)
        draw = ImageDraw.Draw(header_img)

        # Get text bounding box to center it
        naive_bbox = draw.textbbox((0, 0), "Naive", font=font)
        naive_text_width = naive_bbox[2] - naive_bbox[0]
        naive_text_height = naive_bbox[3] - naive_bbox[1]
        
        ours_bbox = draw.textbbox((0, 0), "Ours", font=font)
        ours_text_width = ours_bbox[2] - ours_bbox[0]

        # Calculate positions
        naive_text_x = (base_img_width - naive_text_width) / 2
        ours_text_x = (base_img_width + DIVIDER_WIDTH) + (base_img_width - ours_text_width) / 2
        text_y = (LABEL_SPACE - naive_text_height) / 2
        
        draw.text((naive_text_x, text_y), "Naive", fill=TEXT_COLOR, font=font)
        draw.text((ours_text_x, text_y), "Ours", fill=TEXT_COLOR, font=font)
        
        images_to_stack.append(header_img)

    except Exception as e:
        print(f"CRITICAL: Could not load first image to determine dimensions. Error: {e}")
        return

    # --- 2. Create horizontal stitched rows ---
    for index in INDEXES:
        sample_name = f"sample_{index}"
        print(f"Processing row for {sample_name}...")
        
        try:
            src_naive_path = os.path.join(DIR_NAIVE, sample_name, IMAGE_FILENAME)
            src_ours_path = os.path.join(DIR_OURS, sample_name, IMAGE_FILENAME)
            
            with Image.open(src_naive_path) as img_naive, \
                 Image.open(src_ours_path) as img_ours:
                
                # Check sizes
                if img_naive.size != (base_img_width, base_img_height) or \
                   img_ours.size != (base_img_width, base_img_height):
                    print(f"  Warning: Image size for {sample_name} is incorrect. Skipping.")
                    error_count += 1
                    continue
                
                # Create the horizontal stitched image (no label space)
                h_stitched_img = Image.new("RGB", (total_width, base_img_height), BG_COLOR)
                h_draw = ImageDraw.Draw(h_stitched_img)
                
                # Paste images
                h_stitched_img.paste(img_naive, (0, 0))
                h_stitched_img.paste(img_ours, (base_img_width + DIVIDER_WIDTH, 0))
                
                # Draw divider
                h_draw.line(
                    [(base_img_width, 0), (base_img_width, base_img_height)],
                    fill=DIVIDER_COLOR,
                    width=DIVIDER_WIDTH
                )
                
                images_to_stack.append(h_stitched_img)
                processed_count += 1
        
        except FileNotFoundError as e:
            print(f"  Error: Could not find file for {sample_name}. Skipping.")
            print(f"  Missing file: {e.filename}")
            error_count += 1
        except Exception as e:
            print(f"  An unexpected error occurred for {sample_name}: {e}")
            error_count += 1

    # --- 3. Stitch all images vertically ---
    if processed_count == 0:
        print("No images were successfully processed. Cannot create final image.")
        return

    # Calculate final height: 1 header + (N processed rows * row_height)
    final_height = LABEL_SPACE + (base_img_height * processed_count)
    
    print(f"\nCreating final image with dimensions: {total_width}x{final_height}")
    final_img = Image.new("RGB", (total_width, final_height), BG_COLOR)
    
    current_y = 0
    for img in images_to_stack:
        final_img.paste(img, (0, current_y))
        current_y += img.height
        
    # Save the final image
    final_img.save(FINAL_OUTPUT_FILE)
    
    print("\n==========================================")
    print(f"Successfully created: {FINAL_OUTPUT_FILE}")
    print(f"Total rows stitched:  {processed_count}")
    print(f"Skipped samples:      {error_count}")
    print("==========================================")


if __name__ == "__main__":
    create_vertical_stitch()