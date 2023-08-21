from ultralytics import YOLO
import cv2
import cvzone
import math

# Webcam
# cap = cv2.VideoCapture(0)
cap = cv2.imread('../Videos/fai.tif')

# w_prop_id = cv2.CAP_PROP_FRAME_WIDTH
# h_prop_id = cv2.CAP_PROP_FRAME_HEIGHT
# cap.set(w_prop_id, 1280)
# cap.set(h_prop_id, 720)

model = YOLO('farm.pt')

# Class Names
classNames = ["farm"]

while True:
    # success, img = cap.read()
    cap = cv2.resize(cap, (1280, 720))
    results = model(cap, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            # x1, y1, w, h = box.xywh[0]
            # Converting Tensors to int
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(cap, (x1, y1, w, h))
            # print(x1, y1, x2, y2)
            # cv2.rectangle(img, (x1, y1) , (x2, y2), (255, 0, 255), 3)
            # Confidence
            conf = math.ceil((box.conf[0]*100))/100
            # Class Name
            cls = int(box.cls[0])
            cvzone.putTextRect(cap, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    cv2.imshow("Image", cap)
    cv2.waitKey(0)