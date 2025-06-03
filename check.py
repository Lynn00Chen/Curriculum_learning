import torch
from torch.utils.data import TensorDataset

# Add TensorDataset to the safe globals list for loading
torch.serialization.add_safe_globals([TensorDataset])

def inspect_dataset(file_path):
    """Load and inspect a PyTorch dataset file"""
    print(f"\nInspecting file: {file_path}")
    
    try:
        # Load the data with weights_only=False and map_location='cpu'
        data = torch.load(file_path, weights_only=False, map_location=torch.device('cpu'))
        
        # Check the type of the loaded object
        print(f"Type of loaded object: {type(data)}")
        
        # Handle different potential structures
        if isinstance(data, tuple):
            print(f"This is a tuple with {len(data)} elements")
            for i, element in enumerate(data):
                print(f"  Element {i} type: {type(element)}")
                if isinstance(element, torch.Tensor):
                    print(f"    Shape: {element.shape}")
                    print(f"    Data type: {element.dtype}")
                    print(f"    First few values: {element[0][:5] if element.numel() > 0 else 'Empty tensor'}")
        
        elif isinstance(data, TensorDataset):
            print(f"This is a TensorDataset with {len(data.tensors)} tensors")
            for i, tensor in enumerate(data.tensors):
                print(f"  Tensor {i} shape: {tensor.shape}")
                print(f"  Tensor {i} dtype: {tensor.dtype}")
                print(f"  First few values: {tensor[0][:5] if tensor.numel() > 0 else 'Empty tensor'}")
        
        elif isinstance(data, torch.Tensor):
            print(f"This is a single Tensor")
            print(f"  Shape: {data.shape}")
            print(f"  Data type: {data.dtype}")
            print(f"  First few values: {data[0][:5] if data.numel() > 0 else 'Empty tensor'}")
        
        else:
            print(f"Unrecognized data structure")
            print(f"Content summary: {data}")
            
        return data
    
    except Exception as e:
        print(f"Error loading or inspecting file: {e}")
        return None

# Inspect both files
data1 = inspect_dataset("train_ds.pt")
data2 = inspect_dataset("train_ds1.pt")

# Compare the two datasets
print("\nComparing the two datasets:")
if data1 is not None and data2 is not None:
    # Check if both are the same type
    if type(data1) == type(data2):
        print(f"Both files have the same data structure type: {type(data1)}")
        
        # If they're both tuples, check if they have the same number of elements
        if isinstance(data1, tuple) and isinstance(data2, tuple):
            if len(data1) == len(data2):
                print(f"Both tuples have the same number of elements: {len(data1)}")
                
                # Check if the tensors inside have the same shapes
                all_same_shape = True
                for i in range(len(data1)):
                    if isinstance(data1[i], torch.Tensor) and isinstance(data2[i], torch.Tensor):
                        if data1[i].shape == data2[i].shape:
                            print(f"  Element {i}: Same shape {data1[i].shape}")
                        else:
                            print(f"  Element {i}: Different shapes {data1[i].shape} vs {data2[i].shape}")
                            all_same_shape = False
                
                if all_same_shape:
                    # Check a small sample for content equality
                    try:
                        sample_equal = all(torch.equal(data1[i][:10], data2[i][:10]) for i in range(len(data1)))
                        if sample_equal:
                            print("Sample comparison shows identical content")
                        else:
                            print("Sample comparison shows different content")
                    except:
                        print("Could not compare content samples")
            else:
                print(f"Tuples have different lengths: {len(data1)} vs {len(data2)}")
        
        # If they're TensorDatasets, compare their tensors
        elif isinstance(data1, TensorDataset) and isinstance(data2, TensorDataset):
            if len(data1.tensors) == len(data2.tensors):
                print(f"Both TensorDatasets have the same number of tensors: {len(data1.tensors)}")
                
                for i in range(len(data1.tensors)):
                    if data1.tensors[i].shape == data2.tensors[i].shape:
                        print(f"  Tensor {i}: Same shape {data1.tensors[i].shape}")
                    else:
                        print(f"  Tensor {i}: Different shapes {data1.tensors[i].shape} vs {data2.tensors[i].shape}")
    else:
        print(f"Files have different data structure types: {type(data1)} vs {type(data2)}")
        
        # If one is a TensorDataset and the other is a tuple of tensors, compare the contents
        if (isinstance(data1, TensorDataset) and isinstance(data2, tuple) or 
            isinstance(data2, TensorDataset) and isinstance(data1, tuple)):
            
            tensors1 = data1.tensors if isinstance(data1, TensorDataset) else data1
            tensors2 = data2.tensors if isinstance(data2, TensorDataset) else data2
            
            if len(tensors1) == len(tensors2):
                print(f"Both structures contain the same number of tensors: {len(tensors1)}")
                
                for i in range(len(tensors1)):
                    if tensors1[i].shape == tensors2[i].shape:
                        print(f"  Tensor {i}: Same shape {tensors1[i].shape}")
                    else:
                        print(f"  Tensor {i}: Different shapes {tensors1[i].shape} vs {tensors2[i].shape}")
            else:
                print(f"Different number of tensors: {len(tensors1)} vs {len(tensors2)}")
else:
    print("Could not compare datasets due to loading errors")