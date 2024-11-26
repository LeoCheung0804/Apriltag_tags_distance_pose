import cv2
import numpy as np
import glob

# Define the chessboard size
chessboard_size = (6, 8)
frameSize = (640, 480)

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Size of chessboard squares in millimeters
size_of_chessboard_squares_mm = 25
objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# Get the list of images
images = glob.glob('./images/*.jpg')

def undistort_image(image_path, camera_matrix, dist_coeffs, new_camera_matrix):
        img = cv2.imread(image_path)
        
        if img is None:
            raise FileNotFoundError(f"Image not found or unable to load image at '{image_path}'")

        h, w = img.shape[:2]
        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

        # Undistort
        dst = cv2.undistort(img, camera_matrix, dist_coeffs, None, newCameraMatrix)

        # Crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imwrite('caliResult1.jpg', dst)

        # Undistort with Remapping
        mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, newCameraMatrix, (w, h), 5)
        dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

        # Crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imwrite('caliResult2.jpg', dst)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Check if objpoints and imgpoints are not empty
if objpoints and imgpoints:
    # Perform camera calibration
    ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

    print("Camera Matrix:\n", cameraMatrix)
    #print("Distortion Coefficients:\n", dist)
    #print("Rotation Vectors:\n", rvecs)
    #print("Translation Vectors:\n", tvecs)

    # Save the camera calibration result for later use
    np.savez('calibration_data.npz', cameraMatrix=cameraMatrix, dist=dist, rvecs=rvecs, tvecs=tvecs)

    # Call the function to undistort an image
    #undistort_image('images/cali5.jpg', cameraMatrix, dist, None)

    # Reprojection Error
    mean_error = 0

    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print("total error: {}".format(mean_error / len(objpoints)))
else:
    print("No chessboard corners found in images.")