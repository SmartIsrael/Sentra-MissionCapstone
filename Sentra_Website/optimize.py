#!/usr/bin/env python3
"""
Django Project Image Compressor

This script finds and compresses all images in a Django project to standard web sizes
while maintaining transparency for images that have transparent backgrounds.

Usage:
    python compress_images.py [--max-width 1200] [--quality 80] [--backup] [--exclude-dirs "venv,node_modules"] [--max-size 500]

Options:
    --max-width      Maximum width for images (default: 1200px)
    --quality        JPEG compression quality (0-100, default: 80)
    --backup         Create backups of original images before compressing
    --exclude-dirs   Comma-separated list of directories to exclude
    --max-size       Maximum file size in KB (default: no limit)
"""

import os
import shutil
import argparse
from pathlib import Path
from PIL import Image
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Standard web image file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

# Default directories to exclude
DEFAULT_EXCLUDE_DIRS = {"venv", "env", ".env", ".venv", "node_modules", ".git"}


def get_human_readable_size(size_bytes):
    """Convert bytes to human readable format (KB, MB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compress images in a Django project")
    parser.add_argument(
        "--max-width",
        type=int,
        default=1200,
        help="Maximum width for images (default: 1200px)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=80,
        help="JPEG compression quality (0-100, default: 80)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backups of original images before compressing",
    )
    parser.add_argument(
        "--exclude-dirs",
        type=str,
        default="venv,env,node_modules,.git",
        help="Comma-separated list of directories to exclude",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=500,
        help="Maximum file size in KB (default: 500KB)",
    )

    args = parser.parse_args()

    # Validate quality
    if not (0 <= args.quality <= 100):
        parser.error("Quality must be between 0 and 100")

    return args


def get_exclude_dirs(exclude_dirs_arg):
    """Get the set of directories to exclude."""
    if exclude_dirs_arg:
        custom_excludes = {dir_name.strip() for dir_name in exclude_dirs_arg.split(",")}
        return DEFAULT_EXCLUDE_DIRS.union(custom_excludes)
    return DEFAULT_EXCLUDE_DIRS


def is_excluded_dir(path, exclude_dirs):
    """Check if the path contains any excluded directory."""
    path_parts = Path(path).parts
    return any(exclude_dir in path_parts for exclude_dir in exclude_dirs)


def find_image_files(root_dir, exclude_dirs):
    """Find all image files in the project."""
    image_files = []

    for root, dirs, files in os.walk(root_dir):
        # Skip excluded directories
        if is_excluded_dir(root, exclude_dirs):
            continue

        for filename in files:
            file_path = os.path.join(root, filename)
            file_ext = os.path.splitext(filename)[1].lower()

            # Check if it's an image file by extension
            if file_ext in IMAGE_EXTENSIONS:
                image_files.append(file_path)

    return image_files


def has_transparency(img):
    """Check if the image has transparency."""
    if img.mode == "RGBA":
        # Check if any pixel has an alpha value less than 255
        extrema = img.getextrema()
        if len(extrema) == 4:  # RGBA
            if extrema[3][0] < 255:  # Min alpha value < 255
                return True

    # Also check for 'LA' mode (grayscale with alpha)
    elif img.mode == "LA":
        extrema = img.getextrema()
        if len(extrema) == 2 and extrema[1][0] < 255:
            return True

    return False


def compress_image(image_path, max_width, quality, create_backup, max_size_kb=None):
    """Compress the image while maintaining aspect ratio and transparency if present."""
    try:
        # Get original file size
        original_size = os.path.getsize(image_path)

        # Open image
        img = Image.open(image_path)

        # Create backup if requested
        if create_backup:
            backup_path = f"{image_path}.backup"
            if not os.path.exists(backup_path):
                shutil.copy2(image_path, backup_path)
                logger.info(f"Created backup: {backup_path}")

        # Check original dimensions
        original_width, original_height = img.size

        # Only resize if image width is greater than max_width
        if original_width > max_width:
            # Calculate height to maintain aspect ratio
            ratio = max_width / original_width
            new_height = int(original_height * ratio)

            # Resize image
            img = img.resize((max_width, new_height), Image.LANCZOS)
            logger.info(
                f"Resized {image_path} from {original_width}x{original_height} to {max_width}x{new_height}"
            )

        # Determine output format and parameters
        file_ext = os.path.splitext(image_path)[1].lower()
        is_transparent = has_transparency(img)

        # Start with high quality and gradually reduce until file size is below max_size_kb
        current_quality = quality
        temp_file_path = f"{image_path}.temp"

        while True:
            try:
                # Save with appropriate format and options
                if file_ext in [".jpg", ".jpeg"]:
                    # JPEG doesn't support transparency
                    if img.mode in ["RGBA", "LA"]:
                        img = img.convert("RGB")
                    img.save(temp_file_path, format="JPEG", quality=current_quality, optimize=True)

                elif file_ext == ".png":
                    # Preserve transparency if it exists
                    if is_transparent:
                        # For transparent PNGs, try to reduce colors to save space
                        try:
                            # Use quantization to reduce colors for transparent PNGs
                            # Convert to P mode with transparency
                            img_p = img.convert('RGBA').quantize(colors=256, method=2, kmeans=1, dither=1)
                            img_p.save(temp_file_path, format="PNG", optimize=True, compress_level=9)
                        except Exception:
                            # Fallback to regular save if quantization fails
                            img.save(temp_file_path, format="PNG", optimize=True, compress_level=9)
                    else:
                        # For non-transparent PNGs, convert to JPEG for much better compression
                        img = img.convert("RGB")
                        if max_size_kb and os.path.getsize(image_path) > max_size_kb * 1024:
                            # Large file, convert to JPEG for better compression
                            img.save(temp_file_path, format="JPEG", quality=current_quality, optimize=True)
                        else:
                            # Smaller file, keep as PNG but optimize heavily
                            img.save(temp_file_path, format="PNG", optimize=True, compress_level=9)

                elif file_ext == ".gif":
                    # GIFs might be animated, so we need to handle them differently
                    img.save(temp_file_path, format="GIF", optimize=True)

                elif file_ext == ".webp":
                    # WebP supports transparency
                    if is_transparent:
                        img.save(temp_file_path, format="WEBP", quality=current_quality, method=6)
                    else:
                        img = img.convert("RGB")
                        img.save(temp_file_path, format="WEBP", quality=current_quality, method=6)
                
                # If we get here, saving was successful
            except Exception as e:
                logger.error(f"Error saving temp file for {image_path}: {str(e)}")
                # Clean up and exit
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                return False

            # Check if we've reached the target size
            temp_size = os.path.getsize(temp_file_path)

            # If max_size_kb is specified and the file is still too large
            if max_size_kb and temp_size > max_size_kb * 1024 and current_quality > 20:
                # Reduce quality and try again
                current_quality -= 5
                logger.debug(
                    f"File still too large ({get_human_readable_size(temp_size)}), reducing quality to {current_quality}"
                )
                # Remove temp file before trying again
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                continue

            # Move temp file to original location
            shutil.move(temp_file_path, image_path)
            break

        # After saving, get the new file size
        new_size = os.path.getsize(image_path)
        size_change = original_size - new_size
        percent_saved = (size_change / original_size) * 100 if original_size > 0 else 0

        size_info = f"{get_human_readable_size(original_size)} → {get_human_readable_size(new_size)} (saved {percent_saved:.1f}%)"
        if max_size_kb and new_size > max_size_kb * 1024:
            logger.info(
                f"Compressed: {image_path} - Size: {size_info} - Note: Could not reduce below max size"
            )
        else:
            logger.info(f"Compressed: {image_path} - Size: {size_info}")

        return True

    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        # Clean up temp file if it exists
        if "temp_file_path" in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return False


# This function is already defined above


def main():
    """Main function to compress all images in the Django project."""
    args = parse_arguments()

    # Get current directory (Django project root)
    django_root = os.getcwd()

    # Parse excluded directories
    exclude_dirs = get_exclude_dirs(args.exclude_dirs)

    logger.info(f"Starting image compression in {django_root}")
    logger.info(
        f"Max width: {args.max_width}px, Quality: {args.quality}, Backup: {args.backup}"
    )

    # Find all image files
    image_files = find_image_files(django_root, exclude_dirs)
    logger.info(f"Found {len(image_files)} image files")

    # Process each image
    success_count = 0
    total_original_size = 0
    total_compressed_size = 0
    min_size = float("inf")
    max_size = 0
    min_file = ""
    max_file = ""

    for image_path in image_files:
        original_size = os.path.getsize(image_path)
        total_original_size += original_size

        if compress_image(
            image_path, args.max_width, args.quality, args.backup, args.max_size
        ):
            success_count += 1

            new_size = os.path.getsize(image_path)
            total_compressed_size += new_size

            if new_size < min_size:
                min_size = new_size
                min_file = image_path

            if new_size > max_size:
                max_size = new_size
                max_file = image_path

    # Calculate total savings
    total_saved = total_original_size - total_compressed_size
    percent_saved = (
        (total_saved / total_original_size) * 100 if total_original_size > 0 else 0
    )

    logger.info(
        f"Compression complete. Successfully processed {success_count} out of {len(image_files)} images."
    )
    logger.info(f"Total size before: {get_human_readable_size(total_original_size)}")
    logger.info(f"Total size after: {get_human_readable_size(total_compressed_size)}")
    logger.info(
        f"Total saved: {get_human_readable_size(total_saved)} ({percent_saved:.1f}%)"
    )

    if success_count > 0:
        logger.info(f"Smallest image: {min_file} ({get_human_readable_size(min_size)})")
        logger.info(f"Largest image: {max_file} ({get_human_readable_size(max_size)})")


if __name__ == "__main__":
    main()
# This script is designed to be run from the command line.
