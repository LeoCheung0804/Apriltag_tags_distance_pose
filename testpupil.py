import cv2
import pupil_apriltags as pl
import numpy as np
import os

# Initialize video capture (change 0 to your camera index if needed)
cap = cv2.VideoCapture(1)

# Initialize the AprilTag detector
detector = pl.Detector(families="tag36h11")

# Camera parameters (fx, fy, cx, cy)
fx, fy = 416, 436  # Example focal length in pixels
cx, cy = 327, 375  # Example center point in pixels
camera_params = [fx, fy, cx, cy]
tag_size = 0.077  # Replace with your actual tag size in meters

# Check if the calibration file exists
calibration_file = "calibration_data.npz"
if os.path.exists(calibration_file):
    # Read the camera matrix from the calibration file
    calibration_data = np.load(calibration_file)
    camera_matrix = calibration_data["cameraMatrix"]
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    camera_params = [fx, fy, cx, cy]
else:
    print(
        f"Calibration file {calibration_file} not found. Using default camera parameters."
    )


def show_rotation_matrix(detection):
    """Print the rotation matrix of the detected tag."""
    if hasattr(detection, "pose_R"):
        print(f"Rotation Matrix for tag ID {detection.tag_id}: \n{detection.pose_R}")
    else:
        print(f"Rotation Matrix not available for tag ID {detection.tag_id}")


def calculate_distance_and_differences(tag1, tag2):
    """Calculate the distance and differences between two detected tags."""
    t1, t2 = tag1.pose_t, tag2.pose_t
    distance = np.linalg.norm(t1 - t2) * 100  # Convert to cm
    diff_x, diff_y, diff_z = (t1 - t2) * 100  # Convert to cm
    return distance, diff_x, diff_y, diff_z


def draw_axes(image, camera_params, tag_size, rvec, tvec):
    """Draw the 3D axes on the detected tag."""
    axis_length = tag_size / 2
    axes_points = np.float32(
        [[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]]
    ).reshape(-1, 3)

    fx, fy, cx, cy = camera_params
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    imgpts, _ = cv2.projectPoints(axes_points, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = np.int32(imgpts).reshape(-1, 2)

    origin = tuple(imgpts[0])
    image = cv2.line(image, origin, tuple(imgpts[1]), (0, 0, 255), 3)  # X-axis in red
    image = cv2.line(image, origin, tuple(imgpts[2]), (0, 255, 0), 3)  # Y-axis in green
    image = cv2.line(image, origin, tuple(imgpts[3]), (255, 0, 0), 3)  # Z-axis in blue

    return image


def calculate_distance_to_camera(detection):
    """Calculate the distance from the camera to the detected tag."""
    t = detection.pose_t
    distance = np.linalg.norm(t) * 100  # Convert to cm
    return distance


def display_distance_and_differences(image, results):
    if len(results) >= 2:
        tag1, tag2 = results[0], results[1]
        distance, diff_x, diff_y, diff_z = calculate_distance_and_differences(
            tag1, tag2
        )

        # Display distance and differences on the image
        text = f"Distance: {distance:.3f}cm"
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image, (50, 50 - h), (50 + w, 50 + 5), (0, 255, 255), -1)
        cv2.putText(
            image,
            text,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )

        text = f"dx: {diff_x[0]:.3f}cm, dy: {diff_y[0]:.3f}cm, dz: {diff_z[0]:.3f}cm"
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image, (50, 70 - h), (50 + w, 70 + 5), (0, 255, 255), -1)
        cv2.putText(
            image,
            text,
            (50, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )


while True:
    ret, image = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = detector.detect(
        gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size
    )

    display_distance_and_differences(image, results)

    for r in results:
        tag_id = r.tag_id
        # print(tag_id)  # Print the tag ID

        # Show the rotation matrix
        # show_rotation_matrix(r)

        # Calculate the distance to the camera
        distance_to_camera = calculate_distance_to_camera(r)

        # Print the distance to the camera if 'd' is typed
        # if cv2.waitKey(1) & 0xFF == ord("d"):
        # print(f"Distance to camera for tag ID {tag_id}: {distance_to_camera:.3f}cm")

        # Get the coordinates of the corners
        corners = r.corners.astype(int)
        a, b, c, d = (
            tuple(corners[0]),
            tuple(corners[1]),
            tuple(corners[2]),
            tuple(corners[3]),
        )

        # Draw the detected AprilTag's bounding box
        cv2.line(image, a, b, (255, 0, 255), 2, lineType=cv2.LINE_AA)
        cv2.line(image, b, c, (255, 0, 255), 2, lineType=cv2.LINE_AA)
        cv2.line(image, c, d, (255, 0, 255), 2, lineType=cv2.LINE_AA)
        cv2.line(image, d, a, (255, 0, 255), 2, lineType=cv2.LINE_AA)

        # Draw the center of the AprilTag
        (cx, cy) = (int(r.center[0]), int(r.center[1]))
        cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)  # Draw the center point

        # Annotate the image with the tag ID
        text = str(tag_id)
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image, (cx, cy - h), (cx + w, cy + 5), (0, 255, 255), -1)
        cv2.putText(image, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Annotate the image with the tag family
        tag_family = r.tag_family.decode("utf-8")
        text = f"{tag_family} id:{tag_id}"
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(
            image, (a[0], a[1] - 15 - h), (a[0] + w, a[1] - 15 + 5), (0, 255, 255), -1
        )
        cv2.putText(
            image,
            text,
            (a[0], a[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )

        # Convert rotation matrix to rotation vector
        rvec, _ = cv2.Rodrigues(r.pose_R)
        tvec = r.pose_t

        # Draw the axes for each detected tag
        image = draw_axes(image, camera_params, tag_size, rvec, tvec)

    cv2.imshow("AprilTags", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
