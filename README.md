# TDW Spatial Environment Experiment

A 3D spatial visualization experiment using ThreeDWorld (TDW) to capture multiple camera perspectives of a furnished room scene with field-of-view (FOV) visualization.

## Overview

This project creates a virtual 3D environment containing furniture (sofa and chairs) and captures images from multiple camera positions and orientations. The experiment includes:

- **Main Camera**: Captures natural room views without visual markers
- **Top-down Camera**: Captures overhead views with FOV visualization markers on the ground plane
- **FOV Visualization**: Red markers showing the main camera's field of view area
- **Multiple Perspectives**: 10 different camera positions with 3 random orientations each (30 total image sets)

## Requirements

- Python 3.7+
- ThreeDWorld (TDW) framework
- Required Python packages:
  - `tdw`
  - `pathlib`
  - `math`
  - `random`
  - `os`

## Installation

1. Install TDW following the official documentation
2. Ensure TDW build is available and accessible
3. Install required dependencies:
   ```bash
   pip install tdw
   ```

## Configuration

For running on Linux server, please check: https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/setup/server.md


## Usage

Run the experiment:

```python
python environment.py
```

The script will:
1. Create an empty room with furniture (sofa and two yellow chairs)
2. Set up two cameras (main view and top-down view)
3. Capture 30 image sets from different positions and orientations
4. Save all images to the specified output directory

## Configuration

Key parameters you can modify:

- **Output Directory**: Change `output_directory` path in the main section
- **Camera Positions**: Modify the `camera_positions` list for different viewpoints
- **FOV Settings**: Adjust `MAIN_CAMERA_FOV_ANGLE` (default: 54.43°)
- **Image Count**: Change `num_random_orientations_per_position` for more/fewer views per position

## Output

Images are saved to: `C:\Users\Admin\Desktop\codebase\spatial\images\`

Each image set includes:
- **Main camera view**: Natural room perspective without markers
- **Top-down camera view**: Overhead view with FOV visualization (red markers and blue camera position marker)

Image naming follows TDW's automatic naming convention with timestamps.

## Scene Contents

- **Room**: 12x12 empty room
- **Furniture**:
  - Arflex strips sofa (center, ID: item_id)
  - Two yellow side chairs (positioned at angles, IDs: chair_1_id, chair_2_id)
- **Cameras**:
  - Main camera: 1.6m height, various positions
  - Top-down camera: 10m height, looking down at scene center

## Features

- **FOV Visualization**: Real-time field-of-view markers on ground plane
- **Random Orientations**: Each camera position includes 3 random look-at targets
- **Dual Perspective Capture**: Simultaneous main view and overhead analysis view
- **Automatic Cleanup**: FOV markers are removed after each capture
- **Progress Tracking**: Console output shows capture progress

## Technical Details

- **Image Resolution**: 512x512 pixels
- **Camera FOV**: 54.43° (matches realistic camera settings)
- **FOV Markers**: Small red spheres filling the camera's view area
- **Ground Plane**: Y=0.01 to ensure markers are visible above ground

## Troubleshooting

- Ensure TDW build is running and accessible on the specified port (default: 1071)
- Check that output directory has write permissions
- Verify all required TDW model libraries are available (models_core.json, models_flex.json)
