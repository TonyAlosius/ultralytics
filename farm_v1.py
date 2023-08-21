from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
from shapely.geometry import Polygon
import geopandas as gpd



# Webcam
# cap = cv2.VideoCapture(0)
cap = cv2.imread('../Videos/12.jpeg')

# w_prop_id = cv2.CAP_PROP_FRAME_WIDTH
# h_prop_id = cv2.CAP_PROP_FRAME_HEIGHT
# cap.set(w_prop_id, 1280)
# cap.set(h_prop_id, 720)

model = YOLO('New_25_07_2023.pt')

# Class Names
classNames = ['Dryland', 'Farmland', 'Houses', 'Pond', 'Well']
detections = np.empty((0, 5))
geometries = []
class_ids = []

while True:
    # success, img = cap.read()
    cap = cv2.resize(cap, (1280, 720))
    results = model(cap, stream=True)
    print(results)

    for r in results:
        boxes = r.boxes
        # print("r", r)
        for box in boxes:
            # print(box)
            x1, y1, x2, y2 = box.xyxy[0]
            # print(x1, y1, x2, y2)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)
            w, h = x2-x1, y2-y1
            conf = math.ceil((box.conf[0]*100))/100
            print(conf)
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            print(currentClass)
            cvzone.putTextRect(cap, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                                                      scale=0.6, thickness=1, offset=3)
            cvzone.cornerRect(cap, (x1, y1, w, h), l=5, rt=5)

            # if currentClass == "Pond" and conf > 0.3:
            #     cvzone.putTextRect(cap, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
            #                        scale=0.6, thickness=1, offset=3)
            #     cvzone.cornerRect(cap, (x1, y1, w, h), l=5, rt=5)
            #     currentArray = np.array([x1, y1, x2, y2, cls])
            #     detections = np.vstack((detections, currentArray))
            #     print(detections)
            #     # Shape File Prep
            #     geometry = Polygon([(x1, y1), (x1 + w, y1), (x1 + w, y1 + h), (x1, y1 + h)])
            #     geometries.append(geometry)
            #     class_ids.append(int(cls))
            # print(geometries)

    # # Create a GeoDataFrame
    # gdf = gpd.GeoDataFrame({'geometry': geometries, 'class_id': class_ids}, crs='EPSG:4326')
    # # Step 4: Save the GeoDataFrame to a shapefile
    # output_shapefile = 'segmented_features.shp'
    # gdf.to_file(output_shapefile)

    cv2.imshow("Image", cap)
    cv2.waitKey(0)