import torch
from torch.utils.data import TensorDataset

# Load your data
data = torch.load("processed_data.pt")  # Replace with your actual data path

# Get the sequences
sequences = data
# source_tags = data['source_tags']

# Perform the train/validation split
train_len = round(sequences.shape[0] * 0.8)
train_sequences = sequences[:train_len]
val_sequences = sequences[train_len:]

# # If you want to keep track of the source tags in the split
# train_tags = source_tags[:train_len]
# val_tags = source_tags[train_len:]


# Create input/target pairs for training
train_ds = TensorDataset(train_sequences[:, :-1], train_sequences[:, 1:])
val_ds = TensorDataset(val_sequences[:, :-1], val_sequences[:, 1:])

# train_ds = TensorDataset(train_sequences[:, :-1], train_sequences[:, 1:], train_tags)
# val_ds = TensorDataset(val_sequences[:, :-1], val_sequences[:, s1:], val_tags)

# Save the datasets
torch.save(train_ds, "train_ds1.pt")
torch.save(val_ds, "val_ds1.pt")

print(f"Training dataset with {len(train_ds)} samples saved to train_ds.pt")
print(f"Validation dataset with {len(val_ds)} samples saved to val_ds.pt")