# Tool Development Spec: `k_check_Ascan_gui.py`

## 1. Overview

This document outlines the specification and development plan for a new GUI tool, `k_check_Ascan_gui.py`.

- **Tool Name:** `k_check_Ascan_gui.py`
- **Location:** `tools/visualization/advanced/`
- **Author:** kanda
- **Date:** 2025-07-17

## 2. Purpose

To provide an efficient way to review and analyze a large number of A-scan simulation results (`.out` files) through an interactive GUI. The tool will allow users to quickly navigate between different A-scan files, visualize waveforms, and invoke existing analysis scripts (`k_detect_peak.py`, `k_plot_TWT_estimation.py`) with a single click.

## 3. Specification

### 3.1. Core Functionality

- **Data Input:** The tool will accept a single command-line argument: the path to a JSON file. This JSON file must contain a list of absolute paths to the `.out` files to be reviewed.
- **Auto-loading Configuration:**
    - **zoom_settings.json:** Automatically load zoom_settings.json from the same directory as the input JSON file at startup.
    - **config.json:** Automatically read config.json files corresponding to each .out file (e.g., for `h0.3_w0.3.out`, read `h0.3_w0.3_config.json` from the same directory).
- **GUI Layout:**
    - **A-scan Display:** A central plot area to display the A-scan waveform.
    - **Control Panel:** A dedicated area with buttons and input fields for user interaction.
    - **Information Bar:** A status area showing the current file name and its index in the list (e.g., "File 3 of 50").
- **Interactive Features:**
    - **File Navigation:** Use the left and right arrow keys to switch between previous and next A-scan files in the list.
    - **Zoom Functionality:** 
        - **Time Domain:** Two text boxes in the control panel will allow the user to specify a start and end time (in nanoseconds) to zoom into a specific portion of the A-scan.
        - **Intensity Domain:** Two additional text boxes for specifying minimum and maximum intensity values for amplitude zooming.
        - **Auto-zoom Option:** Checkbox to enable/disable automatic zooming using zoom_settings.json parameters.
        - **Apply Zoom:** Button to refresh the plot with current zoom settings.
    - **Peak Detection:** A "Detect Peaks" button that, when clicked, will call the logic from `tools.analysis.k_detect_peak` and overlay the detected peak locations on the A-scan plot.
    - **TWT Estimation:** A "Show Estimated TWT" button that will call the logic from `tools.visualization.analysis.k_plot_TWT_estimation` and display the result (e.g., as a vertical line) on the plot.
    - **Echo Characteristics Labeling:**
        - **Top Echo:** Three radio buttons for classifying the top echo (Label 1, Label 2, Label 3).
        - **Bottom Echo:** Three radio buttons for classifying the bottom echo (Label 1, Label 2, Label 3).
        - **Save Labels:** Button to save current labeling results to JSON files.

### 3.2. Input Format

- **JSON File Structure:**
  ```json
  {
    "ascan_files": [
      "/path/to/your/simulation/run_001.out",
      "/path/to/your/simulation/run_002.out",
      "/path/to/your/simulation/run_003.out"
    ]
  }
  ```

- **zoom_settings.json Structure:**
  ```json
  {
    "time_start": 0.0,
    "time_end": 10.0,
    "intensity_min": -0.1,
    "intensity_max": 0.1
  }
  ```

- **config.json Structure (per .out file):**
  - Automatically detected and loaded from the same directory as each .out file
  - Naming convention: `{basename}_config.json` where basename is the .out filename without extension
  - Example: For `h0.3_w0.3.out`, the corresponding config file is `h0.3_w0.3_config.json`

### 3.3. Output

- **Primary Output:** The interactive GUI window itself.
- **Labeling Results:** JSON files containing echo characteristics classifications saved to two directory types:
  - `result_use_peak/` directory: For peak-based analysis results
  - `result_use_TWT/` directory: For TWT-based analysis results
  
- **Output JSON Structure:**
  ```json
  {
    "h0.3_w0.3": [0.3, 0.3, 1],
    "h0.3_w0.6": [0.3, 0.6, 1],
    "h0.3_w0.9": [0.3, 0.9, 1],
    "h0.3_w1.2": [0.3, 1.2, 1],
    "h0.3_w1.5": [0.3, 1.5, 1]
  }
  ```
  - Key: Basename of the .out file (without extension)
  - Value: Array of [height, width, label] where label is 1, 2, or 3
  - Separate files generated for Top and Bottom echo classifications

- **Output File Locations:**
  - Based on the input `output_file_paths.json` location
  - Example: For `/path/to/project/output_file_paths.json`:
    - Peak results: `/path/to/project/result_use_peak/`
    - TWT results: `/path/to/project/result_use_TWT/`

### 3.4. Dependencies

- **Python Libraries:** `matplotlib`, `numpy`, `h5py`, `json`
- **Internal gprMax Tools:**
    - `tools.analysis.k_detect_peak`
    - `tools.visualization.analysis.k_plot_TWT_estimation`

**Note:** The successful integration of internal tools depends on their ability to be imported and called as functions. Refactoring of these scripts may be necessary if they are not already modular.

## 4. Development Plan

The development will be broken down into the following steps:

### Step 1: Create Basic GUI Structure and File Loading

- **Task:** Create the initial script `tools/visualization/advanced/k_check_Ascan_gui.py`.
- **Details:**
    - Implement a basic `matplotlib` window.
    - Add command-line argument parsing to accept the input JSON file path.
    - Load the list of file paths from the JSON.
    - Display the first A-scan from the list in the plot window.
    - Set up the basic layout (placeholder for control panel, info bar).

### Step 2: Implement Configuration Auto-loading

- **Task:** Add automatic loading of configuration files.
- **Details:**
    - Implement zoom_settings.json auto-loading from the same directory as input JSON.
    - Add logic to automatically detect and load config.json files for each .out file.
    - Create data structures to store loaded configuration data.
    - Add error handling for missing configuration files.

### Step 3: Implement File Navigation

- **Task:** Enable switching between A-scan files.
- **Details:**
    - Bind the left and right arrow key events to functions that increment/decrement the current file index.
    - Implement the logic to load and redraw the A-scan plot when the file is changed.
    - Update the information bar with the new file name and index.
    - Load corresponding config.json when file changes.

### Step 4: Implement Enhanced Zoom Functionality

- **Task:** Add comprehensive zooming capabilities.
- **Details:**
    - Add "Start Time" and "End Time" `TextBox` widgets for time domain zooming.
    - Add "Min Intensity" and "Max Intensity" `TextBox` widgets for amplitude zooming.
    - Implement auto-zoom checkbox that uses zoom_settings.json parameters.
    - Add "Apply Zoom" `Button` widget to refresh the plot with current zoom settings.
    - Write functions to update both x-axis (time) and y-axis (intensity) limits.

### Step 5: Implement Echo Characteristics Labeling System

- **Task:** Add labeling interface for echo characteristics.
- **Details:**
    - Create radio button groups for Top Echo (3 options) and Bottom Echo (3 options).
    - Add "Save Labels" button to control panel.
    - Implement data structure to track current labeling state for each file.
    - Add logic to determine output directory paths (result_use_peak/, result_use_TWT/).
    - Implement JSON file generation with proper structure and file naming.

### Step 6: Integrate `k_detect_peak.py`

- **Task:** Add the peak detection feature.
- **Details:**
    - **Prerequisite:** Ensure the core logic of `k_detect_peak.py` is encapsulated in a function that can be imported. If not, refactor it first.
    - Add a "Detect Peaks" button to the control panel.
    - When clicked, call the peak detection function with the current A-scan data.
    - Overlay the returned peak coordinates onto the A-scan plot using `matplotlib.pyplot.scatter`.

### Step 7: Integrate `k_plot_TWT_estimation.py`

- **Task:** Add the TWT estimation feature.
- **Details:**
    - **Prerequisite:** Ensure the core logic of `k_plot_TWT_estimation.py` is encapsulated in a function. If not, refactor it first.
    - Add a "Show Estimated TWT" button to the control panel.
    - When clicked, call the TWT estimation function.
    - Draw the returned TWT value on the plot, likely using `matplotlib.pyplot.axvline`.

### Step 8: Final Polish and Testing

- **Task:** Refine the GUI and test all functionalities.
- **Details:**
    - Clean up the layout and add clear labels for all new controls.
    - Test with various valid and invalid inputs (e.g., empty JSON, non-existent file paths, missing config files).
    - Ensure smooth interaction and correct data display.
    - Test labeling system with sample data.
    - Verify JSON output format and file locations.
    - Add comprehensive docstrings and comments to the code.
    - Test zoom functionality with different parameter combinations.
