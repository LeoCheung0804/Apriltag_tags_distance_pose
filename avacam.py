import cv2

def list_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            print ("No camera at index %d" % index)
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr

available_cameras = list_cameras()
print("Available cameras:", available_cameras)