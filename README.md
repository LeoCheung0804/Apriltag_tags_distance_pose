# Main Program

This program calculates the Euclidean distance and differences along the x, y, and z axes between two tags and displays the results on a video frame.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Pupil-Apriltag

## Installation

1. Clone the repository:
    ```sh
    git clone <https://github.com/LeoCheung0804/Apriltag_tags_distance_pose.git>
    cd <Apriltag_tags_distance_pose>
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Parameters

- **Camera Index**: The index of the camera to use for video capture (default is `1`).
- **AprilTag Family**: The family of AprilTags to detect (default is `"tag36h11"`).
- **Camera Parameters**:
  - `fx`: Focal length in pixels along the x-axis (default is `416`).
  - `fy`: Focal length in pixels along the y-axis (default is `436`).
  - `cx`: Principal point x-coordinate in pixels (default is `327`).
  - `cy`: Principal point y-coordinate in pixels (default is `375`).
- **Tag Size**: The size of the AprilTag in meters (default is `0.077`).
- **Calibration File**: The path to the camera calibration file (default is `"calibration_data.npz"`).

## Functions

- **Initialization**:
  - Initializes video capture from the camera.
  - Sets up the AprilTag detector.
  - Reads camera calibration data if available.

- **Tag Detection**:
  - Detects AprilTags in the video frames using the AprilTag detector.
  - Calculates the pose of the detected tags.

- **Distance Calculation**:
  - `calculate_distance_to_camera(detection)`: Calculates the distance from the camera to the detected tag.
  - `calculate_distance_and_differences(tag1, tag2)`: Calculates the distance and differences between two detected tags.

- **Visualization**:
  - `draw_axes(image, camera_params, tag_size, rvec, tvec)`: Draws the 3D axes on the detected tag.
  - `display_distance_and_differences(image, results)`: Displays the calculated distances and differences on the video frames.

## Usage

1. `imagesgen.py`:
   The `imagesgen.py` script captures video from a camera, detects a chessboard pattern in the video frames, and saves images when the 's' key is pressed. The script performs the following steps:

   - Defines the dimensions of the chessboard (default chessboard_size is set to (6, 8)).
   - Initializes video capture from the camera at index 1.

   Key presses:
   - 'q' to quit the loop.
   - 's' to save the current frame as an image in the 'images' directory.

   The script will display the video feed, and you can press 's' to save images or 'q' to quit. The script stops after taking 20 pictures.

2. `calibration.py`:
   The provided code performs camera calibration using images of a chessboard pattern. The calibration process involves detecting chessboard corners in multiple images, computing the camera matrix and distortion coefficients, and then using these parameters to undistort images. The output file is called `calibration_data.npz`.

3. `main.py`:
   - Initializes video capture and AprilTag detector.
   - Detects AprilTags in the video frames.
   - Calculates distances and differences between detected tags.
   - Draws 3D axes on the detected tags and displays the calculated values on the video frames.

## Description

This program calculates the Euclidean distance and differences along the x, y, and z axes between two tags and displays the results on a video frame.

## License

This project is licensed under the MIT License.