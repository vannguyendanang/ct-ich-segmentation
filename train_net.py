import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from data_augmentation import load_and_augment_train_set
import segmentation_models_pytorch as smp
import time
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import WeightedRandomSampler

# Plot training and validation loss curve 
def plot_loss(train_loss, val_loss, epoches):    
    plt.plot(epoches, train_loss, label='Train Loss')
    plt.plot(epoches, val_loss, label='Validation Loss')
    
    plt.xlabel("Epoches")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png', dpi=150, bbox_inches='tight')
    plt.close() # clears the figure before next plot
    # plt.show()
    
# Plot dice score
def plot_dice(dice, epoches, filename, title):
    plt.plot(epoches, dice)
    
    plt.xlabel("Epoches")
    plt.ylabel("Dice")
    plt.title(f"Dice score over epochs for {title}")
    # plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    # plt.show()
    plt.close()  # clears the figure

def plot_dice_for_both(dice_train, dice_val, epoches, filename):
    plt.plot(epoches, dice_train, label='Train dice')
    plt.plot(epoches, dice_val, label='Validation dice')
    
    plt.xlabel("Epoches")
    plt.ylabel("Dice score")
    plt.title("Dice score over epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    # plt.show()
    plt.close()  # clears the figure
    
# Because the number of hemorrhage pixels is very small compared to background pixels in a CT slice (e.g. 4 hemorrhage vs 1000 background), BCE loss is dominated by the majority class. This means the model can "cheat" by predicting everything as background and still achieve a very small loss, since the few missed hemorrhage pixels contribute negligibly to the total.
# Furthermore, BCE evaluates each pixel independently with no awareness of spatial relationships — it only asks "is this pixel right or wrong?" but never asks "are the correct pixels in the right location relative to each other?" This means even if the model predicts the right number of hemorrhage pixels but in the wrong location, BCE loss would still be low. Dice loss addresses this by measuring the overlap between predicted and actual hemorrhage regions — if the predicted pixels are in the wrong location, the intersection is small and Dice loss is large, directly penalizing spatial misalignment.
def combined_loss(preds, masks, pos_weight, alpha=0.5):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        preds, masks, pos_weight=pos_weight, reduction='mean'
    )
    preds_prob = torch.sigmoid(preds)
    
    # Compute Dice per image then average — shape (N,)
    intersection = (preds_prob * masks).sum(dim=(1,2,3))
    dice_per_image = (2 * intersection + 1e-8) / (preds_prob.sum(dim=(1,2,3)) + masks.sum(dim=(1,2,3)) + 1e-8)
    dice_loss = 1 - dice_per_image.mean()
    
    return alpha * bce + (1 - alpha) * dice_loss

def train_model(train_images, train_masks, val_images, val_masks, model, num_epochs=50, batch_size=4, lr=1e-4, pos_weight=1.0, tile_size=512, device=None, patience = 5, min_delta=0.0):
    """
    Train a U-Net model on tiled image data.
    
    Args:
        train_images: numpy array of shape (N, C, H, W) or torch tensor
        train_masks: numpy array of shape (N, H, W) or torch tensor
        model: U-Net model instance
        num_epochs: number of training epochs
        batch_size: batch size
        lr: learning rate
        tile_size: tile size (512 if original images size > 512 such as 1024)
        device: torch device ('cuda' or 'cpu')
    
    Returns:
        trained_model, loss_history
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert to tensors if needed
    if not isinstance(train_images, torch.Tensor):
        train_images = torch.from_numpy(train_images).float()
    if not isinstance(train_masks, torch.Tensor):
        train_masks = torch.from_numpy(train_masks).float()
    
    if not isinstance(val_images, torch.Tensor):
        val_images = torch.from_numpy(val_images).float()
    if not isinstance(val_masks, torch.Tensor):
        val_masks = torch.from_numpy(val_masks).float()

    # Create dataloader
    train_dataset = TensorDataset(train_images, train_masks)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(val_images, val_masks)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Setup optimizer and loss
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_loss_history = []
    val_loss_history = []
    dice_history = []
    avg_dice_pos_history = []
    train_avg_dice_pos_history = []
    best_dice = -1.0
    ret_epoch = 0
    bad_epochs = 0
    
    #  Convert pos_weight into tensor 
    pw = torch.tensor([pos_weight]).to(device)    
    
    for epoch in range(num_epochs):
        # turns on training mode which enables:
        # Dropout — actively dropping neurons randomly
        # Batch Normalization — using the current batch's statistics
        model.train()
        train_epoch_loss = 0.0
        val_epoch_loss = 0.0
        dice = 0.0
        dice_positive = 0.0
        dice_train_positive = 0.0
        positive_count = 0
        train_positive_count = 0
        for images, masks in train_dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            preds = model(images)
              
            # Apply activation function sigmoid on the model output
            preds_act = torch.sigmoid(preds)
            
            # When the model misses a hemorrhage pixel, the loss is 8x larger, which means the gradient is also pos_weight x larger, pushing the model much harder to correct that mistake during backpropagation
            # Large loss → Large gradient → Large weight update
            # Whether weights increase or decrease depends on the direction of the gradient — it could go either way. But the key point is the magnitude of the update is 8x larger for hemorrhage mistakes.
            loss = torch.nn.functional.binary_cross_entropy_with_logits(preds, masks, pos_weight=pw, reduction='mean')
            
            predict_label = (preds_act > 0.5).float() # threshold at 0.5 -> 0,1
            for i in range(images.shape[0]):

                # Dice is undefined/meaningless for all-zero slices
                # Dice gives 0 regardless of whether the model is right or wrong on negative slices — it carries zero information.
                if masks[i].sum() > 0:  # positive slice only
                    # multiply element-wise, count overlapping 1s
                    intersection = (predict_label[i] * masks[i]).sum()
                    # predict_label.sum(): sum number of 1s in this matrix
                    # Add small epsilon to avoid division by zero when no hemorrhage in ground truth
                    score = ((2*intersection)/(predict_label[i].sum() + masks[i].sum() + 1e-8)).item()                    
                    dice_train_positive += score
                    train_positive_count += 1

            loss.backward()
                       
            optimizer.step()
            
            train_epoch_loss += loss.item()
            
        # -- Validation --
        # switches off dropout and batch norm during validation
        model.eval()
        val_predict_label = 0.0
        val_mask = 0.0
        with torch.no_grad():        
            for images, masks in val_dataloader:
                images, masks = images.to(device), masks.to(device)
                pred_val = model(images)
                pred_val_prob = torch.sigmoid(pred_val)
                               
                val_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_val, masks, pos_weight=pw, reduction='mean')                 
                val_epoch_loss += val_loss.item()
                
                # Compute Dice score for each image
                # turn pred_val_prob into label for computing Dice score
                predict_label = (pred_val_prob > 0.5).float() # threshold at 0.5 -> 0,1
                val_predict_label += predict_label.sum().item()
                val_mask += masks.sum().item()
      
                for i in range(images.shape[0]):
                    # multiply element-wise, count overlapping 1s
                    intersection = (predict_label[i] * masks[i]).sum()
                    # predict_label.sum(): sum number of 1s in this matrix
                    # Add small epsilon to avoid division by zero when no hemorrhage in ground truth
                    score = ((2*intersection)/(predict_label[i].sum() + masks[i].sum() + 1e-8)).item()
                    dice += score
                    
                    # Dice is undefined/meaningless for all-zero slices
                    # Dice gives 0 regardless of whether the model is right or wrong on negative slices — it carries zero information.
                    if masks[i].sum() > 0:  # positive slice only
                        dice_positive += score
                        positive_count += 1
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} | LR: {current_lr:.2e} | pred positives: {val_predict_label:.0f} | mask positives: {val_mask:.0f}")       
        train_avg_loss = train_epoch_loss / len(train_dataloader)
        train_loss_history.append(train_avg_loss)
        
        val_avg_loss = val_epoch_loss / len(val_dataloader)
        val_loss_history.append(val_avg_loss)
        
        # Average over total number of possitve images, not batches        
        avg_dice_pos = dice_positive / positive_count if positive_count > 0 else 0.0
        train_avg_dice_pos = dice_train_positive / train_positive_count if train_positive_count > 0 else 0.0
        
        avg_dice_pos_history.append(avg_dice_pos)
        train_avg_dice_pos_history.append(train_avg_dice_pos)
        
        print(f"Avg Dice (positive only):   {avg_dice_pos:.4f}")
        print(f"Avg Dice (positive only) for training set:   {train_avg_dice_pos:.4f}")
        
        #----------EARLY STOPPING LOGIC-----------
        # improvement means: avg_dice_pos > best_dice + min_delta
        if avg_dice_pos > best_dice + min_delta:
            best_dice = avg_dice_pos
            ret_epoch = epoch + 1
            bad_epochs = 0
            # saves the model's learned weights to a file called best_model.pth
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            bad_epochs += 1
        
        # stop if no improvement for 'patience' epochs
        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch + 1}. Best dice score ={best_dice:.3f}. Best epoch={ret_epoch}")
            break
    
    return train_loss_history, val_loss_history, train_avg_dice_pos_history, avg_dice_pos_history, ret_epoch

# ===========================================
# Read and load data
ct_scans_dir = Path("/home/tnguye47/Data/physionet.org/files/ct-ich/material/data/image")
masks_dir = Path("/home/tnguye47/Data/physionet.org/files/ct-ich/material/data/label")

# load train and validation data set
train_df = pd.read_csv('./data/train_split.csv')
val_df   = pd.read_csv('./data/val_split.csv')

# Check imbalance from original CSV before augmentation
total = len(train_df)
hemorrhage = train_df[train_df['No_Hemorrhage'] == 0].shape[0]
no_hemorrhage = train_df[train_df['No_Hemorrhage'] == 1].shape[0]

print(f"Total slices in training set: {total}")
print(f"Slices with hemorrhage: {hemorrhage} ({hemorrhage/total*100:.1f}%)")
print(f"Slices without hemorrhage: {no_hemorrhage} ({no_hemorrhage/total*100:.1f}%)")
   
# Run data augmentation for training set
X_train, y_train = load_and_augment_train_set(
    train_df, 
    ct_scans_dir, 
    masks_dir,
    num_augmentations=3  # creates 3 augmented versions per slice
)

print(f"Original train slices:   {len(train_df)}")
print(f"After augmentation:      {len(X_train)}")  # ~4x more
print(f"X_train shape:           {X_train.shape}")  # (N, 512, 512)
print(f"y_train shape:           {y_train.shape}")  # (N, 512, 512)

# Read input and labels for validation set
X_val = []
y_val = []

for _, row in val_df.iterrows():
    patient_no = int(row['PatientNumber'])
    slice_no   = int(row['SliceNumber'])

    # Load using patient + slice number
    ct_slice   = np.array(Image.open(ct_scans_dir / f"{patient_no}_{slice_no}.png"))
    mask_slice = np.array(Image.open(masks_dir  / f"{patient_no}_{slice_no}.png"))

    # Convert mask back to 0/1
    mask_slice = (mask_slice / 255).astype(np.uint8)

    # Add original
    ct_slice_expanded = np.expand_dims(ct_slice, axis=0)  # (1, 512, 512)
    X_val.append(ct_slice_expanded)
      
    mask_slice_expanded = np.expand_dims(mask_slice, axis=0)  # (1, 512, 512)
    y_val.append(mask_slice_expanded)    
    
# Convert the list to a numpy array so it can be converted to a PyTorch tensor, 
# which is required as input to the model
X_val = np.array(X_val)
y_val = np.array(y_val)

# Initialize U Net model
# in_channels=1:  The input has one channel, which makes sense for the CT scans since they are grayscale images. 
# num_classes = 1: 1 channel in the output. That single channel contains a matrix of probability values, one for each pixel. 
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1
)

# Normalize input data before training
X_train = X_train.astype(np.float32) / 255.0
X_val = X_val.astype(np.float32) / 255.0

# Check input data before train
print(f"X_train dtype: {X_train.dtype}, min: {X_train.min()}, max: {X_train.max()}")
print(f"y_train dtype: {y_train.dtype}, min: {y_train.min()}, max: {y_train.max()}")
print(f"X_val dtype:   {X_val.dtype},   min: {X_val.min()},   max: {X_val.max()}")
print(f"y_val dtype:   {y_val.dtype},   min: {y_val.min()},   max: {y_val.max()}")
              
# Check validation data
print(f"Total val slices loaded: {len(X_val)}")
print(f"Positive val slices loaded: {(y_val.sum(axis=(1,2,3)) > 0).sum()}")
print(f"Negative val slices loaded: {(y_val.sum(axis=(1,2,3)) == 0).sum()}")
print(f"Total positive pixels in val: {y_val.sum()}")

# Check imbalance at pixel level
# sum over channel+ H + W → one value per slice
# result = [1000, 0, 500]   # shape: (3,) — one value per slice, so slice 1 has 1000 pixel 1, slice 2 has 0 pixel 1, slice 3 has 500 pixel 1
total_pixels = y_train.sum(axis=(1,2,3))  
pos_pixels = total_pixels.sum()           # total hemorrhage pixels
# y_train.size = 3 × 1 × 512 × 512 = 786,432   # total pixels
neg_pixels = y_train.size - pos_pixels    # total background pixels
# pos_weight = neg_pixels / pos_pixels
pos_weight = min(neg_pixels / pos_pixels, 5.0)
print(f"Pixel-level pos_weight: {pos_weight:.1f}")  # expect 50–200x

# Start train model
start = time.time()

# train_loss_history, dice_history, avg_dice_pos_history, ret_epoch \
train_loss_history, val_loss_history, dice_train_history, avg_dice_pos_history, ret_epoch \
= train_model(X_train, y_train, X_val, y_val, model, num_epochs=200, batch_size=16, lr=1e-5, pos_weight=pos_weight, tile_size=512, device=None, patience = 30, min_delta=0.001 
)

end = time.time()
elapsed = end - start
print(f"Training took {elapsed/60:.2f} minutes")

epochs = range(1, len(train_loss_history) + 1)
plot_loss(train_loss_history,val_loss_history, epochs)
plot_dice_for_both(dice_train_history,avg_dice_pos_history, epochs,"dice_pos_curse.png")

print("Loss and dice plots are saved!")