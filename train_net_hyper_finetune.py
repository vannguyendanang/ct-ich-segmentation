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
import optuna

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
def plot_dice(dice, epoches, filename):
    plt.plot(epoches, dice)
    
    plt.xlabel("Epoches")
    plt.ylabel("Dice")
    plt.title("Dice score over epochs")
    # plt.legend()
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

    # create a mask has True/False values
    # Sample value of pos_mask: [True, False, True, False]
    # Shape of pos_mask = num of images in the batch
    pos_mask = masks.sum(dim=(1,2,3)) > 0
    
    # if there is no possitive slices, just return normal loss without consideration of dice score because this is purely loss for the negative slices
    if pos_mask.sum() == 0:
        return bce
    
    # select only slice that is True, Shape = (num of possitive slices, 1, 512, 512)
    pos_pred = preds[pos_mask]
    pos_for_mask = masks[pos_mask]
    preds_prob = torch.sigmoid(pos_pred)
    # Compute Dice per image then average — shape (N,)
    intersection = (preds_prob * pos_for_mask).sum(dim=(1,2,3))
    dice_pos_images = (2 * intersection) / (preds_prob.sum(dim=(1,2,3)) + pos_for_mask.sum(dim=(1,2,3)) + 1e-8)
    dice_loss = 1 - dice_pos_images.mean()

    return alpha * bce + (1 - alpha) * dice_loss

def train_model(train_images, train_masks, val_images, val_masks, model, num_epochs=50, batch_size=4, lr=1e-4, pos_weight=1.0, tile_size=512, device=None, patience = 5, min_delta=0.0, alpha=0.5):
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
    # shuffle=False for validation data
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Setup optimizer and loss
    model = model.to(device)
    
    # Adam applies weight decay through the gradient, which gets distorted by Adam's adaptive learning rate — the regularization effect becomes inconsistent
    # AdamW applies weight decay directly to the weights, keeping regularization clean and separate from the gradient update
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,  weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    
    train_loss_history = []
    val_loss_history = []
    dice_history = []
    avg_dice_pos_history = []
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
        positive_count = 0
        for images, masks in train_dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            preds = model(images)
              
            # When the model misses a hemorrhage pixel, the loss is 8x larger, which means the gradient is also pos_weight x larger, pushing the model much harder to correct that mistake during backpropagation
            # Large loss → Large gradient → Large weight update
            # Whether weights increase or decrease depends on the direction of the gradient — it could go either way. But the key point is the magnitude of the update is 8x larger for hemorrhage mistakes.
            loss = combined_loss(preds, masks.float(), pw, alpha=alpha)
            
            loss.backward()
            
            # add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
                              
                val_loss = combined_loss(pred_val, masks.float(), pw, alpha=alpha)
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
        train_avg_loss = train_epoch_loss / len(train_dataloader)
        train_loss_history.append(train_avg_loss)
        
        val_avg_loss = val_epoch_loss / len(val_dataloader)
        val_loss_history.append(val_avg_loss)
        
        # Average over total number of positive images, not batches
        avg_dice_pos = dice_positive / positive_count if positive_count > 0 else 0.0
        scheduler.step()
        
        avg_dice_pos_history.append(avg_dice_pos)
        
        avg_dice = dice/len(val_dataloader.dataset)
        dice_history.append(avg_dice)
        
        #----------EARLY STOPPING LOGIC-----------
        # improvement means: avg_dice_pos > best_dice + min_delta
        if avg_dice_pos > best_dice + min_delta:
            best_dice = avg_dice_pos
            ret_epoch = epoch + 1
            bad_epochs = 0
        else:
            bad_epochs += 1
        
        # stop if no improvement for 'patience' epochs
        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch + 1}. Best dice score ={best_dice:.3f}. Best epoch={ret_epoch}")
            break
    
    return train_loss_history, val_loss_history, dice_history, avg_dice_pos_history, ret_epoch


# ==========================================
def objective(trial):
    # Step 1: Optuna suggests values
    lr         = trial.suggest_float('lr', 3e-4, 2e-3, log=True)
    pos_weight = trial.suggest_float('pos_weight', 2.0, 10.0) #min = 2.0, max=10.0
    alpha      = trial.suggest_float('alpha', 0.3, 0.7) #min = 0.3, max=0.7
    
    # Build fresh model
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1
    )
    
    # Run training with suggested params
    train_loss_history, val_loss_history, dice_history, avg_dice_pos_history, ret_epoch = train_model(
        X_train, y_train, X_val, y_val,
        model,
        num_epochs=200,
        batch_size=16,
        lr=lr,                  
        pos_weight=pos_weight,  
        patience=30,
        min_delta=0.001,
        alpha = alpha
    )
    
    # Return best Dice achieved in this run
    best_dice = max(avg_dice_pos_history)
    return best_dice  # Optuna tries to maximize this
    
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

# Start train model
start = time.time()

study = optuna.create_study(direction='maximize')  # maximize Dice
study.optimize(objective, n_trials=10)             # run 10 experiments

end = time.time()
elapsed = end - start
print(f"Training took {elapsed/3600:.2f} hours")

# Get all trials as a dataframe
trials_df = study.trials_dataframe()

# Select and rename relevant columns
results = trials_df[['number', 'value', 'params_lr', 'params_pos_weight', 'params_alpha']]
results.columns = ['Trial', 'Best Dice', 'LR', 'Pos Weight', 'Alpha']

# Sort by best Dice descending
results = results.sort_values('Best Dice', ascending=False).reset_index(drop=True)

# Round for cleaner display
results['Best Dice'] = results['Best Dice'].round(4)
results['LR'] = results['LR'].apply(lambda x: f'{x:.2e}')  # e.g. 1.23e-04
results['Pos Weight'] = results['Pos Weight'].round(2)
results['Alpha'] = results['Alpha'].round(3)

print(results.to_string(index=False))

#  highlight the best trial
print(f"\nBest Trial: {study.best_trial.number}")
print(f"Best Dice:  {study.best_value:.4f}")
print(f"Best Params:")
for key, val in study.best_params.items():
    print(f"  {key}: {val}")