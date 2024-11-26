import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

num = 0

def generate_chessboard_image():
    chessboard_size = (9, 6)
    
    # Calculate square size to fit the desired resolution
    desired_width = 1920
    desired_height = 1080
    square_size = min(desired_width // chessboard_size[0], desired_height // chessboard_size[1])

    img_size = (chessboard_size[0] * square_size, chessboard_size[1] * square_size)
    chessboard_img = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)

    for i in range(chessboard_size[1]):
        for j in range(chessboard_size[0]):
            if (i + j) % 2 == 0:
                cv2.rectangle(chessboard_img, (j * square_size, i * square_size),
                              ((j + 1) * square_size, (i + 1) * square_size), 255, -1)

    return chessboard_img

while cap.isOpened():

    succes, img = cap.read()

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('images/img' + str(num) + '.png', img)
        print("image saved!")
        num += 1
        if num == 20:
            break
    elif k == ord('c'): # wait for 'c' key to generate chessboard image
        chessboard_img = generate_chessboard_image()
        plt.imsave('images/chessboard' + '.pdf', chessboard_img, cmap='gray')
        print("chessboard image saved as PDF!")

    cv2.imshow('Img', img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release and destroy all windows before termination
cap.release()
cv2.destroyAllWindows()