import numpy as np

# Load the NPZ file with allow_pickle set to True
data = np.load('RAW-20230425-132107084.npz', allow_pickle=True)

# List all arrays in the NPZ file
print("Arrays in the NPZ file:", data.files)

# Inspect the contents of each array
for key in data.files:
    print(f"\nArray name: {key}")
    array = data[key]
    if isinstance(array, np.ndarray):
        print("Shape:", array.shape)
        print("Data type:", array.dtype)
        print("First few values:", array[:5] if array.size > 5 else array)
    else:
        print("Data type:", type(array))
        print("Value:", array)
