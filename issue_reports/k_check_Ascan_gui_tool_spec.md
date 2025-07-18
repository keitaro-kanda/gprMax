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
- **GUI Layout:**
    - **A-scan Display:** A central plot area to display the A-scan waveform.
    - **Control Panel:** A dedicated area with buttons and input fields for user interaction.
    - **Information Bar:** A status area showing the current file name and its index in the list (e.g., "File 3 of 50").
- **Interactive Features:**
    - **File Navigation:** Use the left and right arrow keys to switch between previous and next A-scan files in the list.
    - **Zoom Functionality:** Two text boxes in the control panel will allow the user to specify a start and end time (in nanoseconds) to zoom into a specific portion of the A-scan. An "Apply Zoom" button will refresh the plot.
    - **Peak Detection:** A "Detect Peaks" button that, when clicked, will call the logic from `tools.analysis.k_detect_peak` and overlay the detected peak locations on the A-scan plot.
    - **TWT Estimation:** A "Show Estimated TWT" button that will call the logic from `tools.visualization.analysis.k_plot_TWT_estimation` and display the result (e.g., as a vertical line) on the plot.

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

### 3.3. Output

- The primary output is the interactive GUI window itself.
- No file-based output is required in the initial version.

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

### Step 2: Implement File Navigation

- **Task:** Enable switching between A-scan files.
- **Details:**
    - Bind the left and right arrow key events to functions that increment/decrement the current file index.
    - Implement the logic to load and redraw the A-scan plot when the file is changed.
    - Update the information bar with the new file name and index.

### Step 3: Implement Zoom Functionality

- **Task:** Add interactive zooming.
- **Details:**
    - Add "Start Time" and "End Time" `TextBox` widgets and an "Apply Zoom" `Button` widget to the control panel.
    - Write the function that reads the values from the text boxes and updates the x-axis limits of the A-scan plot.

### Step 4: Integrate `k_detect_peak.py`

- **Task:** Add the peak detection feature.
- **Details:**
    - **Prerequisite:** Ensure the core logic of `k_detect_peak.py` is encapsulated in a function that can be imported. If not, refactor it first.
    - Add a "Detect Peaks" button to the control panel.
    - When clicked, call the peak detection function with the current A-scan data.
    - Overlay the returned peak coordinates onto the A-scan plot using `matplotlib.pyplot.scatter`.

### Step 5: Integrate `k_plot_TWT_estimation.py`

- **Task:** Add the TWT estimation feature.
- **Details:**
    - **Prerequisite:** Ensure the core logic of `k_plot_TWT_estimation.py` is encapsulated in a function. If not, refactor it first.
    - Add a "Show Estimated TWT" button to the control panel.
    - When clicked, call the TWT estimation function.
    - Draw the returned TWT value on the plot, likely using `matplotlib.pyplot.axvline`.

### Step 6: Final Polish and Testing

- **Task:** Refine the GUI and test all functionalities.
- **Details:**
    - Clean up the layout and add clear labels.
    - Test with various valid and invalid inputs (e.g., empty JSON, non-existent file paths).
    - Ensure smooth interaction and correct data display.
    - Add docstrings and comments to the code.
