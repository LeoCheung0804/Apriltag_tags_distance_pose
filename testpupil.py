import cv2
import pupil_apriltags as pl
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture(0)  # Change 0 to your camera index if needed

# Initialize the detector
detector = pl.Detector(families="tag36h11")

# Camera calibration parameters (example values, you need to calibrate your camera)
camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

def draw_axes(image, corners, rvec, tvec, camera_matrix, dist_coeffs):
    axis_length = 0.05  # Length of the axes in meters
    axis_points = np.float32([
        [axis_length, 0, 0], 
        [0, axis_length, 0], 
        [0, 0, axis_length]
    ]).reshape(-1, 3)

    imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = imgpts.astype(int)

    corner = tuple(corners[0].ravel())
    image = cv2.line(image, corner, tuple(imgpts[0].ravel()), (0, 0, 255), 5)
    image = cv2.line(image, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    image = cv2.line(image, corner, tuple(imgpts[2].ravel()), (255, 0, 0), 5)
    return image

while True:
    ret, image = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = detector.detect(gray)

    for r in results:
        tag_id = r.tag_id
        print(tag_id)  # Print the tag ID

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
        cv2.putText(
            image, str(tag_id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

        # Annotate the image with the tag family
        tag_family = r.tag_family.decode("utf-8")
        cv2.putText(
            image,
            f"{tag_family} id:{tag_id}",
            (a[0], a[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        # Estimate pose of the AprilTag
        rvec, tvec, _ = cv2.estimateAffine3D(corners, np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32))

        # Draw axes
        image = draw_axes(image, corners, rvec, tvec, camera_matrix, dist_coeffs)

    cv2.imshow('AprilTag Detection', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
