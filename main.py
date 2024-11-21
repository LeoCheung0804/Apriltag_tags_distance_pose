import cv2
import apriltag
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture(0)  # Change 0 to your camera index if needed

# Specify the tag family
options = apriltag.DetectorOptions(families="tag36h11")
detector = apriltag.Detector(options)

# Camera parameters (fx, fy, cx, cy)
fx, fy = 416, 436  # Example focal length in pixels
cx, cy = 327, 375  # Example center point in pixels
camera_params = (fx, fy, cx, cy)
tag_size = 0.077  # Replace with your actual tag size in meters

def draw_axes(frame, pose, camera_params, tag_size):
    fx, fy, cx, cy = camera_params
    axis_length = tag_size / 2  # Length of the axes in meters

    # Define the axes in the tag's coordinate system
    axes = np.float32([
        [0, 0, 0],  # Origin
        [axis_length, 0, 0],  # X-axis
        [0, axis_length, 0],  # Y-axis
        [0, 0, -axis_length]  # Z-axis
    ]).reshape(-1, 3)

    # Project the axes points to the image plane
    rvec, _ = cv2.Rodrigues(pose[:3, :3])
    tvec = pose[:3, 3]
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    imgpts, _ = cv2.projectPoints(axes, rvec, tvec, camera_matrix, None)

    # Draw the axes
    origin = tuple(imgpts[0].ravel().astype(int))
    cv2.line(frame, origin, tuple(imgpts[1].ravel().astype(int)), (0, 0, 255), 2)  # X-axis in red
    cv2.line(frame, origin, tuple(imgpts[2].ravel().astype(int)), (0, 255, 0), 2)  # Y-axis in green
    cv2.line(frame, origin, tuple(imgpts[3].ravel().astype(int)), (255, 0, 0), 2)  # Z-axis in blue

def annotate_tag_with_id(frame, detection, font, font_scale, font_thickness, text_color, bg_color):
    tag_id = str(detection.tag_id)
    (text_width, text_height), baseline = cv2.getTextSize(tag_id, font, font_scale, font_thickness)
    text_x, text_y = int(detection.center[0]), int(detection.center[1]) - 10

    # Draw the background rectangle
    cv2.rectangle(frame, (text_x, text_y - text_height - baseline), (text_x + text_width, text_y + baseline), bg_color, cv2.FILLED)

    # Draw the text
    cv2.putText(frame, tag_id, (text_x, text_y), font, font_scale, text_color, font_thickness)

def main():
    bg_color = (0, 255, 255)  # Yellow background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 2
    text_color = (0, 0, 0)  # Black text

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(gray)

        if len(detections) >= 2:
            # Estimate pose of each detected tag
            pose1, e0_1, e1_1 = detector.detection_pose(detections[0], camera_params, tag_size)
            pose2, e0_2, e1_2 = detector.detection_pose(detections[1], camera_params, tag_size)

            # Extract translation vectors
            t1 = pose1[:3, 3]
            t2 = pose2[:3, 3]

            # Calculate Euclidean distance
            distance = np.linalg.norm(t1 - t2) * 100  # Convert to centimeters

            # Calculate differences along x, y, and z axes
            x_diff = (t2[0] - t1[0]) * 100  # Convert to centimeters
            y_diff = (t2[1] - t1[1]) * 100  # Convert to centimeters
            z_diff = (t2[2] - t1[2]) * 100  # Convert to centimeters

            # Print the differences on the frame
            text = f"Distance: {distance:.3f} cm, X diff: {x_diff:.3f} cm, Y diff: {y_diff:.3f} cm, Z diff: {z_diff:.3f} cm"
            # Define text properties
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 2
            text_color = (0, 0, 0)  # Black text
            bg_color = (0, 255, 255)  # Yellow background

            # Get the text size
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

            # Define the text position
            text_x, text_y = 10, 30

            # Draw the background rectangle
            cv2.rectangle(frame, (text_x, text_y - text_height - baseline), (text_x + text_width, text_y + baseline), bg_color, cv2.FILLED)

            # Draw the text
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness)

        for detection in detections:
            # Estimate pose of each detected tag
            pose, e0, e1 = detector.detection_pose(detection, camera_params, tag_size)

            # Draw the detection
            for idx in range(len(detection.corners)):
                cv2.line(frame, tuple(detection.corners[idx - 1, :].astype(int)),
                         tuple(detection.corners[idx, :].astype(int)), (0, 255, 0), 2)

            # Draw the center
            cv2.circle(frame, tuple(detection.center.astype(int)), 5, (0, 0, 255), -1)

            # Draw the axes
            draw_axes(frame, pose, camera_params, tag_size)

            # Annotate the tag with its ID
            annotate_tag_with_id(frame, detection, font, font_scale, font_thickness, text_color, bg_color)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

