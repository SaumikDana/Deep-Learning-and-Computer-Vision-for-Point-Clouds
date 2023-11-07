__author__ = "Saumik Dana"
__date__ = "10/31/2023"
__purpose__ = "Checking contents of npz file(s)"

import numpy as np
import os

def main():

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    file_path = os.path.join(script_dir, '..', 'tests', 'data', 'mannequin', 'REC-20230425-132107084.npz')

    print("Absolute Path:", file_path)

    if not os.path.exists(file_path):
        print("Error: File does not exist at the specified path.")
    else:
        print("File exists.")

    # Load the NPZ file with allow_pickle set to True
    data = np.load(file_path, allow_pickle=True)

    # List all arrays in the NPZ file
    print("Arrays in the NPZ file:", data.files)

    # Inspect the contents of each array
    for key in data.files:
        print(f"\nArray name: {key}")
        array = data[key]
        if isinstance(array, np.ndarray):
            print("Shape:", array.shape)
            print("Data type:", array.dtype)
            try:
                print("First few values:", array[:5] if array.size > 5 else array)
            except IndexError:
                print("Value:", array)
        else:
            print("Data type:", type(array))
            print("Value:", array)

if __name__ == "__main__":
    main()
