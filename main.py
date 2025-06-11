import cv2
from ultralytics import YOLO
import pandas as pd
import cvzone
import numpy as np
from test1 import process_frame
import os
from tracker import*
from datetime import datetime
# Add these imports for immediate license plate processing
from process_license_images import create_database, detect_license_plate_from_image, insert_license_record


# Load vehicle detection model (replace with your vehicle model)
vehicle_model = YOLO("yolov10s.pt") 
# Load license plate detection model
lp_model = YOLO("best.pt")  

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('videos/VID_20250430_111055.mp4')  # Replace with your video file path
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
tracker=Tracker()
count = 0
#area = [(397, 326), (449, 387), (816, 370), (633, 312)] # for pixel7
area = [(237, 356), (428, 512), (891, 428), (604, 365)] # for VID...

# Create directory for today's date
today_date = datetime.now().strftime('%Y-%m-%d')
output_dir = os.path.join('saved_images', today_date)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
license_plate_dir = os.path.join('license_plates', today_date)
os.makedirs(license_plate_dir, exist_ok=True)

list1=[]
conn, cursor = create_database()
while True:
    ret, frame = cap.read()
    count += 1
    if count % 2 != 0:
        continue
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 600))
    processed_frame, detected_label = process_frame(frame)
    print(detected_label)
    results = vehicle_model(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list=[]
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        
        d = int(row[5])
        c = class_list[d]
        list.append([x1, y1, x2, y2])
    bbox_idx = tracker.update(list)
    for bbox in bbox_idx:
        x3, y3, x4, y4, id = bbox
    
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
#        cv2.circle(frame,(cx,cy),4,(255,0,0),-1)
#        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
        result = cv2.pointPolygonTest(np.array(area, np.int32), ((cx, cy)), False)
        if result >= 0:
           if 'car' in c and detected_label == "RED":
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                
                # Save the image with red label
                timestamp = datetime.now().strftime('%H-%M-%S')
                image_filename = f"{timestamp}.jpg"
                output_path = os.path.join(output_dir, image_filename)
                if list1.count(id)==0:
                   list1.append(id)
                   cv2.imwrite(output_path, frame)

                   # --- License Plate Detection on Saved Violation Frame ---
                   violation_img = cv2.imread(output_path)
                   lp_results = lp_model(violation_img)
                   for i, lp_box in enumerate(lp_results[0].boxes.xyxy):
                        x_lp1, y_lp1, x_lp2, y_lp2 = map(int, lp_box)
                        lp_crop = violation_img[y_lp1:y_lp2, x_lp1:x_lp2]
                        lp_filename = f"{timestamp}_lp_{i}.jpg"
                        lp_path = os.path.join(license_plate_dir, lp_filename)
                        cv2.imwrite(lp_path, lp_crop)
                        print(f"Saved license plate crop: {lp_path}")

                        license_plate = detect_license_plate_from_image(lp_path, "Red Light Violation")
                        if license_plate:
                            insert_license_record(cursor, conn, license_plate, "Red Light Violation", lp_path)

           else:     
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)


    cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 255, 0), 2)

    cv2.imshow("RGB", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(4) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Close the database connection after processing
conn.close()

# After video processing is complete, automatically process license plate images
print("\n" + "="*80)
print("VIDEO PROCESSING COMPLETED!")
print("="*80)
print(f"Check the database for license plate records from {today_date}")