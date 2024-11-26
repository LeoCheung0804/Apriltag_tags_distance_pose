import cv2
import os

# Ensure the 'images' directory exists
if not os.path.exists('images'):
    os.makedirs('images')

# Define the dimensions of the chessboard
chessboard_size = (6, 8)  # Adjust the pattern size according to your chessboard

# Define the criteria for the corner sub-pixel accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Start capturing video
cap = cv2.VideoCapture(1)

num = 0

while cap.isOpened():
    ret1, frame = cap.read()
    if not ret1:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Break the loop on 'q' key press
        break
    elif cv2.waitKey(1) == ord('s'):  # wait for 's' key to save and exit
        name = os.path.join('images', 'img' + str(num) + '.jpg')
        cv2.imwrite(name, frame)
        print(f"{name} saved!")
        num += 1
        if num == 20:
            break

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        # Refine the corner positions
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Draw the chessboard corners
        cv2.drawChessboardCorners(frame, chessboard_size, corners2, ret)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()