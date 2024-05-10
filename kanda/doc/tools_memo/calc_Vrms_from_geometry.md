This Python code defines a class called calc_Vrms_from_geometry that calculates the root-mean-square velocity (Vrms) from geological layer properties specified in a JSON file. Below is a detailed description of the code.

## Imports
- numpy: A library for numerical computations, used here for array operations and mathematical functions.
- json: A library for handling JSON data, used here to read input parameters from a JSON file.

## Class Definition: calc_Vrms_from_geometry
This class is designed to calculate Vrms from the geometric properties of geological layers.

## Constructor
Parameters:
- jsonfile_path: The path to the JSON file containing the layer properties.
- c: The speed of light in vacuum (default value is $3 \times 10 ^ 8$ m/s).

### Attributes:
- self.jsonfile_path: Stores the path to the JSON file.
- self.c: Stores the speed of light value.

## Method: load_params_from_json
- Functionality: Loads layer properties from a JSON file and computes the internal velocity of each layer.

### Returns:
- self.layer_thickness: An array of layer thicknesses (in meters).
- self.internal_permittivity: An array of the permittivity of each layer.
- self.internal_velovity: An array of the internal velocities of each layer (calculated using the permittivity and the speed of light).

## Method: calc_t0
### Parameters:
- layer_thickness: An array of layer thicknesses.
- internal_velovity: An array of internal velocities.
- Functionality: Calculates the two-way travel time (t0) for each layer.
### Returns:
An array of t0 values for each layer.

## Method: calc_Vrms
### Parameters:
- layer_thickness: An array of layer thicknesses.
- internal_velocity: An array of internal velocities.
- t0: An array of two-way travel times.
- Functionality: Calculates the Vrms for each layer.

### Returns:
An array of Vrms values for each layer.

