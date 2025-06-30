import cv2
from ultralytics import YOLO
import pandas as pd
import cvzone
import numpy as np
from test1 import process_frame
import os
from tracker import*
from datetime import datetime
import send_sms_beem
# Add these imports for immediate license plate processing
from process_license_db import (
    create_database, 
    detect_license_plate_from_image, 
    insert_license_record, 
    get_owner_info_by_license, 
    sync_vehicle_owners_with_detected_plates
)

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

cap = cv2.VideoCapture('videos/pixel7.mp4')  # Replace with your video file path
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
tracker=Tracker()
count = 0
area = [(397, 326), (449, 387), (816, 370), (633, 312)] # for pixel7
#area = [(237, 356), (428, 512), (891, 428), (604, 365)] # for VID...

# Create directory for today's date
today_date = datetime.now().strftime('%Y-%m-%d')
output_dir = os.path.join('saved_images', today_date)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
license_plate_dir = os.path.join('license_plates', today_date)
os.makedirs(license_plate_dir, exist_ok=True)

# Initialize tracking lists
list1 = []  # Track processed vehicle IDs
sms_sent_list = []  # Track license plates that have been sent SMS
processed_license_plates = set()  # Track processed license plates to avoid duplicates

# Create database connection
conn, cursor = create_database()
if conn is None or cursor is None:
    print("❌ Failed to connect to database. Exiting.")
    exit(1)

# Ensure vehicle_owners table is synced with any existing license records
print("🔄 Syncing vehicle owners with existing license records...")
sync_vehicle_owners_with_detected_plates(cursor, conn)
print("✅ Database sync completed!")

print(f"🎬 Starting video processing for {today_date}")
print("🚦 Monitoring for red light violations...")
print("-" * 60)

while True:
    ret, frame = cap.read()
    count += 1
    if count % 2 != 0:
        continue
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 600))
    processed_frame, detected_label = process_frame(frame)
    
    # Only print traffic light status when it changes
    if count % 20 == 0:  # Print every 20 frames to reduce console spam
        print(f"Traffic Light Status: {detected_label}")
    
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
        
        result = cv2.pointPolygonTest(np.array(area, np.int32), ((cx, cy)), False)
        
        if result >= 0:
            if 'car' in c and detected_label == "RED":
                cvzone.putTextRect(frame, f'VIOLATION-{id}', (x3, y3), 1, 1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                
                # Save the image with red label
                timestamp = datetime.now().strftime('%H-%M-%S-%f')[:-3]  # Include milliseconds
                image_filename = f"violation_{timestamp}_id{id}.jpg"
                output_path = os.path.join(output_dir, image_filename)
                
                if list1.count(id) == 0:
                    list1.append(id)
                    cv2.imwrite(output_path, frame)
                    
                    print(f"\n🚨 RED LIGHT VIOLATION DETECTED!")
                    print(f"Vehicle ID: {id}")
                    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"Image saved: {image_filename}")

                    # --- License Plate Detection on Saved Violation Frame ---
                    violation_img = cv2.imread(output_path)
                    lp_results = lp_model(violation_img)
                    
                    violation_processed = False
                    
                    if len(lp_results[0].boxes.xyxy) > 0:
                        print(f"🔍 Detected {len(lp_results[0].boxes.xyxy)} license plate(s)")
                        
                        for i, lp_box in enumerate(lp_results[0].boxes.xyxy):
                            x_lp1, y_lp1, x_lp2, y_lp2 = map(int, lp_box)
                            lp_crop = violation_img[y_lp1:y_lp2, x_lp1:x_lp2]
                            lp_filename = f"lp_{timestamp}_id{id}_{i}.jpg"
                            lp_path = os.path.join(license_plate_dir, lp_filename)
                            cv2.imwrite(lp_path, lp_crop)
                            print(f"💾 Saved license plate crop: {lp_filename}")

                            # Detect license plate text using OCR
                            license_plate = detect_license_plate_from_image(lp_path, "Red Light Violation")
                            
                            if license_plate and license_plate not in processed_license_plates:
                                processed_license_plates.add(license_plate)
                                
                                # Insert into database (this also creates owner record if not exists)
                                if insert_license_record(cursor, conn, license_plate, "Red Light Violation", output_path):
                                    violation_processed = True
                                    print(f"📝 License plate '{license_plate}' added to database")
                                    
                                    # Send SMS if not already sent for this license plate
                                    if license_plate not in sms_sent_list:
                                        print(f"📞 Preparing to send SMS for license plate: {license_plate}")
                                        
                                        # Get owner information from database
                                        owner_info = get_owner_info_by_license(cursor, license_plate)
                                        
                                        if owner_info:
                                            owner_name = owner_info['owner_name']
                                            phone_number = owner_info['phone_number']
                                            
                                            print(f"👤 Owner: {owner_name}")
                                            print(f"📱 Phone: {phone_number}")
                                            print("📤 Sending SMS notification...")
                                            
                                            # Send SMS using the violation SMS function
                                            try:
                                                sms_success = send_sms_beem.send_violation_sms(
                                                    phone_number=phone_number,
                                                    owner_name=owner_name,
                                                    license_plate=license_plate,
                                                    area="traffic intersection"
                                                )
                                                
                                                if sms_success:
                                                    print(f"✅ SMS sent successfully to {owner_name} ({phone_number})")
                                                    sms_sent_list.append(license_plate)
                                                    
                                                    # Add visual indicator on frame
                                                    cvzone.putTextRect(frame, f'SMS SENT!', (x3, y3-60), 1, 1, colorR=(0, 255, 0))
                                                else:
                                                    print(f"❌ Failed to send SMS to {owner_name} ({phone_number})")
                                                    cvzone.putTextRect(frame, f'SMS FAILED!', (x3, y3-60), 1, 1, colorR=(255, 0, 0))
                                                    
                                            except Exception as e:
                                                print(f"❌ SMS sending error: {e}")
                                                cvzone.putTextRect(frame, f'SMS ERROR!', (x3, y3-60), 1, 1, colorR=(255, 0, 0))
                                        else:
                                            print(f"⚠️ No owner information found for license plate: {license_plate}")
                                            cvzone.putTextRect(frame, f'NO OWNER INFO!', (x3, y3-60), 1, 1, colorR=(255, 255, 0))
                                    else:
                                        print(f"📱 SMS already sent for license plate: {license_plate}")
                                        cvzone.putTextRect(frame, f'SMS ALREADY SENT', (x3, y3-60), 1, 1, colorR=(0, 255, 255))
                            elif license_plate:
                                print(f"🔄 License plate '{license_plate}' already processed")
                    else:
                        print("⚠️ No license plates detected in violation image")
                        cvzone.putTextRect(frame, f'NO LP DETECTED', (x3, y3-30), 1, 1, colorR=(255, 255, 0))
                    
                    # Add violation indicator
                    cvzone.putTextRect(frame, f'VIOLATION!', (x3, y3-30), 1, 1, colorR=(0, 0, 255))
                    print("-" * 40)

            else:     
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)

    # Draw the monitoring area
    cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 255, 0), 2)
    
    # Add status information to frame
    status_text = f"Light: {detected_label} | Violations: {len(list1)} | SMS Sent: {len(sms_sent_list)}"
    cvzone.putTextRect(frame, status_text, (10, 30), 1, 1, colorR=(0, 0, 0))

    cv2.imshow("RGB", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(4) & 0xFF == ord('q'):
        print("\n🛑 User requested to stop video processing...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Close the database connection after processing
conn.close()

# After video processing is complete, display comprehensive summary
print("\n" + "="*80)
print("🎬 VIDEO PROCESSING COMPLETED!")
print("="*80)
print(f"📅 Date: {today_date}")
print(f"🚗 Total vehicles with violations: {len(list1)}")
print(f"🔢 Unique license plates processed: {len(processed_license_plates)}")
print(f"📱 SMS notifications sent: {len(sms_sent_list)}")

if sms_sent_list:
    print(f"\n📋 License plates with SMS notifications sent:")
    for i, plate in enumerate(sms_sent_list, 1):
        print(f"   {i}. {plate}")

if processed_license_plates:
    print(f"\n📋 All processed license plates:")
    for i, plate in enumerate(processed_license_plates, 1):
        print(f"   {i}. {plate}")

print(f"\n💾 Images saved to: {output_dir}")
print(f"🔍 License plate crops saved to: {license_plate_dir}")
print(f"🗄️ Database records updated for date: {today_date}")
print("="*80)
print("✅ System shutdown complete!")