import albumentations as A
import numpy as np
from PIL import Image

# Define augmentation pipeline
# wraps all augmentations into one pipeline, applying them all together 
# to each image in both CT image and mask simultaneously
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5), #50% chance applied, 50% chance skipped
    
    # `limit=10` → rotates randomly between -10° and +10°
    #  p=0.5 — probability of applying this augmentation
    A.Rotate(limit=10, p=0.5),
    
    A.RandomBrightnessContrast(
        brightness_limit=0.2, # ± 20% brightness change
        contrast_limit=0.2, # ± 20% contrast change
        p=0.5  #  probability of applying this augmentation
    ),
    
    A.RandomResizedCrop(
        height=512, 
        width=512, 
        scale=(0.8, 1.0),  # zoom between 80% and 100%
        p=0.5
    ),
    
    # A.ElasticTransform(
    #     alpha=1,        # intensity of deformation
    #     sigma=10,       # smoothness of deformation
    #     # alpha_affine=50,
    #     p=0.3
    # ),
])

# Apply augmentation to CT slice + mask together
def augment_slice(ct_slice, mask_slice):
    """
    ct_slice:   2D numpy array (512x512) - windowed CT image
    mask_slice: 2D numpy array (512x512) - binary mask (0 or 1)
    """
    # CT slice after windowing is already 0-255 but stored as float64
    # Albumentations requires uint8 format (0-255 integers)
    ct_uint8   = ct_slice.astype(np.uint8)
    # mask_uint8 = (mask_slice * 255).astype(np.uint8)
    mask_uint8 = mask_slice.astype(np.uint8)
    
    # Apply same augmentation to both
    # Returns a dictionary with keys 'image' and 'mask'
    augmented   = augmentation(image=ct_uint8, mask=mask_uint8)
    
    ct_aug   = augmented['image']
    # mask_aug = (augmented['mask'] / 255).astype(np.uint8)  # back to 0/1
    mask_aug = augmented['mask']
    
    return ct_aug, mask_aug


# Apply to entire train set
def load_and_augment_train_set(train_df, images_dir, masks_dir, num_augmentations=3):
    X_train = []
    y_train = []
    
    for _, row in train_df.iterrows():
        patient_no = int(row['PatientNumber'])
        slice_no   = int(row['SliceNumber'])
        has_hemorrhage = int(row['No_Hemorrhage']) == 0  # 0 means HAS hemorrhage
        
        # Load using patient + slice number
        ct_slice   = np.array(Image.open(images_dir / f"{patient_no}_{slice_no}.png"))
        mask_slice = np.array(Image.open(masks_dir  / f"{patient_no}_{slice_no}.png"))
        
        # Convert mask back to 0/1
        mask_slice = (mask_slice / 255).astype(np.uint8)
             
        # Add original images
        # At channel for ct_slice because U-Net expects the input in (N, C, H, W) format 
        ct_slice_expanded = np.expand_dims(ct_slice, axis=0)  # (1, 512, 512)
        X_train.append(ct_slice_expanded)
        
        mask_slice_expanded = np.expand_dims(mask_slice, axis=0)  # (1, 512, 512)
        y_train.append(mask_slice_expanded)
        
        # Augment ONLY hemorrhage slices
        if has_hemorrhage:
            for _ in range(num_augmentations):
                ct_aug, mask_aug = augment_slice(ct_slice, mask_slice)

                # expand the channel size into ct slice
                ct_aug = np.expand_dims(ct_aug, axis=0) # (1, 512, 512)
                X_train.append(ct_aug)

                mask_slice_expanded = np.expand_dims(mask_aug, axis=0)  # (1, 512, 512)
                y_train.append(mask_slice_expanded)
    
    return np.array(X_train), np.array(y_train)
