import h5py

def print_keys(obj, indent=0):
    if isinstance(obj, h5py.File) or isinstance(obj, h5py.Group):
        for key in obj.keys():
            print('  ' * indent + key)
            print_keys(obj[key], indent + 1)
    elif isinstance(obj, h5py.Dataset):
        print('  ' * indent + obj.name)

h5_file = 'output/output_data.h5'
with h5py.File(h5_file, 'r') as f:
    print_keys(f)
