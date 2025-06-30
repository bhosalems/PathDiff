import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool

def has_color(img):
    """Check if an RGB image has more than just grayscale colors."""
    # If the image has less than 3 channels, it's not RGB, so we treat it as not having color.
    if img.shape[2] < 3:
        return False
    
    # A color image in RGB should have at least some pixels where R, G, and B are not all the same.
    # This checks for any such pixels.
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            r, g, b = img[i, j]
            if r != g or g != b or b != r:
                return True
    return False

def process_image(args):
    index, sample = args
    img_path = '/a2il/data/mbhosale/PathDiff/QUILT_1M/resized/quilt_1m/' + sample['image_path']
    if not os.path.exists(img_path):
        return (index, 'not_found')
    try:
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        if img_np is None or not has_color(img_np):
            return (index, 'lacks_color')
    except Exception as e:
        return (index, 'error')
    return None

def process_image2(args):
    index, sample = args
    img_path = '/a2il/data/mbhosale/PathDiff/QUILT_1M/resized/quilt_1m/' + sample['image_path']
    if not os.path.exists(img_path):
        return (index, 'not_found')
    try:
        with Image.open(img_path) as img:
            if img_path.lower().endswith('.jpg') or img_path.lower().endswith('.jpeg'):
                if img.mode == 'RGB':
                    return (index, 'jpeg_with_3_channels')
                else:
                    return (index, 'jpeg_not_3_channels')
            img = img.convert('RGBA')
            if img.mode == 'RGBA' and img_path.lower().endswith('.png'):
                return (index, 'png_with_4_channels')
            elif img.mode == 'RGBA' and not img_path.lower().endswith('.png'):
                return (index, 'non_png_with_4_channels')
            else:
                return (index, 'other')
    except Exception as e:
        return (index, 'error')

def validate_preprocess(lookup):
    with Pool(processes=os.cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_image2, lookup.iterrows()), total=len(lookup), desc="Conforming images"))

    # Process results to categorize images
    categories = {
        'png_with_4_channels': [],
        'jpeg_with_3_channels': [],
        'jpeg_not_3_channels': [],
        'non_png_with_4_channels': [],
        'other': [],
        'not_found': [],
        'error': []
    }

    for result in results:
        if result:
            categories[result[1]].append(result[0])

    for category, indices in categories.items():
        print(f"{category}: {len(indices)} images")
    return categories

if __name__ == '__main__':
    # lookup = pd.read_csv('/a2il/data/mbhosale/PathDiff/QUILT_1M/resized/quilt_1M_lookup.csv', index_col=0)
    # with Pool(processes=os.cpu_count()) as pool:
    #     results = list(tqdm(pool.imap(process_image, lookup.iterrows()), total=len(lookup), desc="Processing images"))

    # # Process results to get indices
    # img_not_found_indices = [result[0] for result in results if result and result[1] == 'not_found']
    # lacks_color_indices = [result[0] for result in results if result and result[1] == 'lacks_color']
    # # The lacks_color_indices list will now have the indices of images that should be considered for deletion.

    # # Combine and drop indices
    # drop_indices = set(img_not_found_indices + lacks_color_indices)

    # lookup.drop(index=drop_indices, inplace=True)
    # lookup.to_csv('/a2il/data/mbhosale/PathDiff/QUILT_1M/resized/quilt_1M_lookup_preprocessed.csv', index=False)      

    lookup_preprocessed = pd.read_csv('/a2il/data/mbhosale/PathDiff/QUILT_1M/resized/quilt_1M_lookup_preprocessed.csv')
    # Now validate whether the filtering was done correctly.    
    categories = validate_preprocess(lookup_preprocessed)
