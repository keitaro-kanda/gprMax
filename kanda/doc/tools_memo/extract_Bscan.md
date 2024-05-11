# Tool Description: Extracting Observation Data from B-scan Data

This tool is used to extract observation data from pre-existing B-scan data and create a new B-scan. The input mergedout file must be B-scan observation data.

## Imports
- `numpy`: A library for numerical computations, used for array operations and mathematical functions.
- `json`: A library for handling JSON data, used to read input parameters.
- `os`: A library for interacting with the operating system, used to manage file paths.
- `h5py`: A library for working with HDF5 files, used for storing and managing large amounts of data.
- `tools.outputfiles_merge`: A custom module used to get output data.
- `argparse`: A library for parsing command-line arguments.

## Command-Line Argument Parsing
The script uses `argparse` to handle command-line arguments.

```python
# Parse command line arguments
parser = argparse.ArgumentParser(usage='cd gprMax; python -m tools.extract_Bscan jsonfile')
parser.add_argument('jsonfile', help='json file path')
args = parser.parse_args()
```

## Load JSON Data
The script loads JSON data from the file specified by the command-line argument.
```python
# Load JSON data
with open(args.jsonfile) as f:
    params = json.load(f)
```

## Load Original B-scan Data
The script loads the original B-scan data using the get_output_data function from the custom tools.outputfiles_merge module.

```python
# Load original B-scan data
original_data_path = params['original_info']['original_out_file']
data, dt = get_output_data(original_data_path, 1, 'Ez')
print(dt)
print(data.shape)
```

## Original B-scan Antenna Parameters
The script retrieves antenna parameters from the original B-scan data.

```python
# Antenna parameters of original B-scan data
with open(params['original_info']['original_json_file']) as original_json:
    original_params = json.load(original_json)
original_src_step = original_params['antenna_settings']['src_step'] # [m]
original_rx_step = original_params['antenna_settings']['rx_step'] # [m]
original_src_move_times = original_params['antenna_settings']['src_move_times']
```

## Extracted B-scan Antenna Parameters
The script retrieves antenna parameters for the extracted B-scan.

```python
# Antenna parameters after extraction
extracted_src_start = params['antenna_settings']['src_start'] # [m]
extracted_src_step = params['antenna_settings']['src_step'] # [m]
extracted_rx_start = params['antenna_settings']['rx_start'] # [m]
extracted_rx_step = params['antenna_settings']['rx_step'] # [m]
extracted_src_move_times = params['antenna_settings']['src_move_times']
```

## Antenna Step Ratio Calculation
The script calculates the antenna step ratio for the extraction process.

```python
# Extraction antenna step ratio (must be integer and >= 1)
src_step_ratio = int(extracted_src_step / original_src_step) # Example: extracted_src_step = 2 [m], original_src_step = 1 [m], src_step_ratio = 2
rx_step_ratio = int(extracted_rx_step / original_rx_step)
```

## Class Definition: extract_Bscan
This class is designed to extract B-scan data based on the given parameters.

## Constructor (__init__)
### Parameters:
- data: The original B-scan data.

### Attributes:
- self.data: Stores the original B-scan data.

### Method: extract_Bscan
- Parameters:
    - src_move_times: The number of times the source moves.
- Functionality: Extracts the B-scan data based on the source move times and step ratio.
- Returns: An array of the extracted B-scan data.

### Method: output_Bscan
- Parameters:
    - outputfile: The path to the output file.
- Functionality: Outputs the extracted B-scan data to an HDF5 file.
- Returns: The HDF5 file object.

```python
# Extract B-scan data
class extract_Bscan:
    def __init__(self, data):
        self.data = data

    def extract_Bscan(self, src_move_times):
        self.Bscan = np.zeros((self.data.shape[0], src_move_times))
        for i in range(src_move_times):
            self.Bscan[:, i] = (self.data[:, src_step_ratio * i])
        return np.array(self.Bscan)

    def output_Bscan(self, outputfile):
        output_dir = os.path.dirname(args.jsonfile)
        with h5py.File(outputfile, 'w') as f:
            f.create_dataset(output_dir + '/extracted_Bscan', data=self.Bscan)
        return f
```

## Example Usage
The example demonstrates how to use the extract_Bscan class to extract and save B-scan data.

```python
# Usage
extracted_Bscan = extract_Bscan(data)
extracted_Bscan = extracted_Bscan.extract_Bscan(extracted_src_move_times)
print('extracted_Bscan shape: ', extracted_Bscan.shape)

# Output extracted B-scan data
output_dir = os.path.dirname(args.jsonfile)
np.savetxt(output_dir + '/extracted_Bscan.txt', extracted_Bscan)
```