# This code loads the CT slices (grayscale images) of the brain-window for each subject in ct_scans folder then saves them to
# one folder (data\image).
# Their segmentation from the masks folder is saved to another folder (data\label).

import os
from pathlib import Path
import numpy as np
import pandas as pd
# from scipy.misc import imread, imresize, imsave
from skimage.io import imread, imsave
from skimage.transform import resize
import nibabel as nib
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm

def imresize(arr, size):
    return resize(arr, size, preserve_range=True).astype(np.uint8)

def window_ct (ct_scan, w_level=40, w_width=120):
    w_min = w_level - w_width / 2
    w_max = w_level + w_width / 2
#     Assume ct_scan.shape = (512, 512, 30),--> grabs the 3rd dimension → 30 slices
    num_slices=ct_scan.shape[2]
    for s in range(num_slices):
#         extracts one 2D size (512×512) slice at a time
        slice_s = ct_scan[:,:,s]
# Linearly scales HU values to 0-255 range:
# At w_min (-20 HU) → pixel value = 0 (black)
# At w_max (100 HU) → pixel value = 255 (white)    
        slice_s = (slice_s - w_min)*(255/(w_max-w_min)) #or slice_s = (slice_s - (w_level-(w_width/2)))*(255/(w_width))
# Clips values outside the window — anything below -20 HU (0 pixel) becomes 0 (black), anything above 100 HU (255 pixel) becomes 255 (white) because:
#  + image format (PNG, JPG) only support 0-255.    
#  + neural networks train better with normalized inputs
        slice_s[slice_s < 0]=0
        slice_s[slice_s > 255] = 255
        #slice_s=np.rot90(slice_s)
        ct_scan[:,:,s] = slice_s

    return ct_scan

def extract_img_label_each_slice():
    numSubj = 82
    new_size = (512, 512)
    window_specs=[40,120] #Brain window
    currentDir = Path(os.getcwd())
    datasetDir = str(Path(currentDir))

    # Reading labels
    hemorrhage_diagnosis_df = pd.read_csv(
        Path(currentDir, 'hemorrhage_diagnosis_raw_ct.csv'))
    # convert ds into numpy array
    # hemorrhage_diagnosis_array = hemorrhage_diagnosis_df._get_values
    hemorrhage_diagnosis_array = hemorrhage_diagnosis_df.values

    # reading images
    train_path = Path('data')
    image_path = train_path / 'image'
    label_path = train_path / 'label'
    if not train_path.exists():
        train_path.mkdir()
        image_path.mkdir()
        label_path.mkdir()
    else:
        # ct-scan and masks are already generated, no need further process
        return image_path, label_path

    for sNo in tqdm(range(0+49, numSubj+49), desc="Processing patients"):
            
        if sNo>58 and sNo<66: #no raw data were available for these subjects
            next
        else:
            #Loading the CT scan
            ct_dir_subj = Path(datasetDir,'ct_scans', "{0:0=3d}.nii".format(sNo))
            ct_scan_nifti = nib.load(str(ct_dir_subj))
            # ct_scan = ct_scan_nifti.get_data()
            ct_scan = ct_scan_nifti.get_fdata()
            # Linearly scales HU values to 0-255 range, ct_scan will have shape 512 x 512 x num slices
            ct_scan = window_ct(ct_scan, window_specs[0], window_specs[1])

            #Loading the masks
            masks_dir_subj = Path(datasetDir,'masks', "{0:0=3d}.nii".format(sNo))
            masks_nifti = nib.load(str(masks_dir_subj))
    # The mask is a 3D matrix with only 0 and 1 values. 0 means background, no hemorrhage, 1 means hemorrhage (raw mask)         
            # masks = masks_nifti.get_data()
            masks = masks_nifti.get_fdata()
    # idx is is an array of 2814 True/False values to find all rows belonging to the current patient
            idx = hemorrhage_diagnosis_array[:, 0] == sNo
    # retrieves all slice numbers for the current patient   
            sliceNos = hemorrhage_diagnosis_array[idx, 1]
    # retrieves all NoHemorrhage for the current patient      
            NoHemorrhage = hemorrhage_diagnosis_array[idx, 7]
            if sliceNos.size!=ct_scan.shape[2]:
                print('Warning: the number of annotated slices does not equal the number of slices in NIFTI file!')

            for sliceI in range(0, sliceNos.size):
                # Saving the a given CT slice
                               
                # extracts one slice → shape (H, W)
                x = imresize(ct_scan[:,:,sliceI], new_size)
                    
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)                
                    # imsave saves x as a grayscale PNG → no channel dimension added
                    imsave(image_path / f"{sNo}_{sliceNos[sliceI]}.png", x)
                
                mask_slice = np.round(masks[:,:,sliceI]).astype(np.uint8)    
                x = imresize(mask_slice, new_size)
                              
                # clamps all values to be within 0-255 range and convert to int
                x = np.clip(x, 0, 255).astype(np.uint8)
                   
                # Suppress low contrast warning for binary mask images
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)                
                    imsave(label_path / f"{sNo}_{sliceNos[sliceI]}.png", x)
                
    return image_path, label_path

ct_scans_dir, masks_dir = extract_img_label_each_slice()