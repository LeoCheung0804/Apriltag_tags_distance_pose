import cv2
import pupil_apriltags as pl
import numpy

# Initialize video capture
cap = cv2.VideoCapture(0)  # Change 0 to your camera index if needed

# Initialize the detector
detector = pl.Detector(families="tag36h11")

def show_rotation_matrix(detection):
    if hasattr(detection, 'pose_R'):
        print("Rotation Matrix for tag ID {}: \n{}".format(detection.tag_id, detection.pose_R))
    else:
        print("Rotation Matrix not available for tag ID {}".format(detection.tag_id))

while True:
    ret, image = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = detector.detect(gray, estimate_tag_pose=True)

    for r in results:
        tag_id = r.tag_id
        print(tag_id)  # Print the tag ID

        # Show the rotation matrix
        show_rotation_matrix(r)

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

    cv2.imshow("AprilTags", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
