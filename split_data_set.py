import pandas as pd
from sklearn.model_selection import train_test_split

# Read and load data
ds_hemorrhage = pd.read_csv("hemorrhage_diagnosis_raw_ct.csv")

# get unique patient numbers
all_patients = ds_hemorrhage['PatientNumber'].unique()
# get one label per patient
all_labels = ds_hemorrhage.groupby('PatientNumber')['No_Hemorrhage'].min().values

# split the training and test set
train_val_patients, test_patients, train_val_labels, test_labels = train_test_split(
    all_patients,
    all_labels,
    test_size=0.10,
    random_state=42,
    # shuffle=True,
    stratify=all_labels  # ensures ~same ratio of hemorrhage/healthy in each split
)


# split the training and validation set
train_patients, val_patients = train_test_split(
    train_val_patients,
    test_size=0.30,
    random_state=42,
    # shuffle=True, # shuffle = True already the default value
    stratify=train_val_labels  # ensures ~same ratio of hemorrhage/healthy in each split
)

# get slices for each split
train_df = ds_hemorrhage[ds_hemorrhage['PatientNumber'].isin(train_patients)]
val_df = ds_hemorrhage[ds_hemorrhage['PatientNumber'].isin(val_patients)]
test_df = ds_hemorrhage[ds_hemorrhage['PatientNumber'].isin(test_patients)]

train_df.to_csv('./data/train_split.csv', index=False)
val_df.to_csv('./data/val_split.csv', index=False)
test_df.to_csv('./data/test_split.csv', index=False)

print(f"Train slices:      {len(train_df)}")
print(f"Validation slices: {len(val_df)}")
print(f"Test slices:       {len(test_df)}")