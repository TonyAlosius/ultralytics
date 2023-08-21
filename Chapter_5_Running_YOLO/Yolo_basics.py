from ultralytics import YOLO
import cv2

model = YOLO('../Yolo_Weights/yolov8l.pt')
results = model("Images/3.png", show=True)
# 0 means until the user stops
cv2.waitKey(0)