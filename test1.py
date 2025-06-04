import cv2
import numpy as np

def process_frame(frame):
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Define the region of interest (ROI) where traffic lights are expected
    # These coordinates should be adjusted based on your specific video
    roi_x_min, roi_y_min = 313, 50  # Top-left corner for pixel7
    roi_x_max, roi_y_max = 462, 218  # Bottom-right corner for pixel7

    #roi_x_min, roi_y_min = 144, 92  # Top-left corner for VID..
    #roi_x_max, roi_y_max = 308, 242  # Bottom-right corner for VID..
    
    # Ensure ROI is within frame boundaries
    roi_x_min = max(0, min(roi_x_min, width-1))
    roi_y_min = max(0, min(roi_y_min, height-1))
    roi_x_max = max(roi_x_min+1, min(roi_x_max, width))
    roi_y_max = max(roi_y_min+1, min(roi_y_max, height))
    
    # Extract the ROI from the frame
    roi_frame = frame[roi_y_min:roi_y_max, roi_x_min:roi_x_max].copy()
    
    # Check if ROI is valid
    if roi_frame.size == 0:
        print("Warning: ROI is empty. Using full frame instead.")
        roi_frame = frame.copy()
        roi_x_min, roi_y_min = 0, 0
        roi_x_max, roi_y_max = width, height
    
    # Draw ROI rectangle on the frame (for visualization)
    cv2.rectangle(frame, (roi_x_min, roi_y_min), (roi_x_max, roi_y_max), (255, 255, 0), 1)
    
    # Define the color ranges
    # Green traffic light range - expanded range to catch more green shades
    lower_green = np.array([30, 80, 80])  # Lower threshold to catch more green variations
    upper_green = np.array([90, 255, 255])  # Higher threshold to include yellowish-green
    
    # Red traffic light ranges - improved values
    # Primary red range (low hue values)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    
    # Secondary red range (high hue values)
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

    # Create masks for color ranges
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Combine the two red masks
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Apply morphological operations to clean up the masks
    kernel = np.ones((3, 3), np.uint8)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.dilate(mask_green, kernel, iterations=1)  # Dilate to connect nearby green pixels
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    
    # Combine green and red masks
    combined_mask = cv2.bitwise_or(mask_green, mask_red)
    
    # Threshold the combined mask
    _, final_mask = cv2.threshold(combined_mask, 254, 255, cv2.THRESH_BINARY)
    detected_label = None

    # Find contours
    cnts, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    valid_traffic_lights = []
    
    for c in cnts:
        # Filter by contour area - relaxed thresholds for green lights
        area = cv2.contourArea(c)
        if area < 30 or area > 1500:  # Relaxed thresholds
            continue
            
        x, y, w, h = cv2.boundingRect(c)
        
        # Filter by aspect ratio - relaxed for green lights
        aspect_ratio = h / float(w)
        if aspect_ratio < 0.8:  # Relaxed threshold
            continue
            
        # Calculate the center point of the rectangle
        cx = x + w // 2
        cy = y + h // 2
        
        # Determine the color of the contour
        green_pixels = cv2.countNonZero(mask_green[y:y+h, x:x+w])
        red_pixels = cv2.countNonZero(mask_red[y:y+h, x:x+w])
        
        # Debug information (commented out for production use)
        # print(f"Contour at ({x},{y}): green pixels={green_pixels}, red pixels={red_pixels}")
        
        if green_pixels > 0:  # Green color range
            color = (0, 255, 0)  # Green color for the rectangle
            text_color = (0, 255, 0)  # Green text
            label = "GREEN"
            # Add a higher confidence for green to prioritize it when both colors are detected
            confidence = min(area / 500.0, 1.0) * (aspect_ratio if aspect_ratio < 3 else 3/aspect_ratio) * 1.2
        elif red_pixels > 0:  # Red color range
            color = (0, 0, 255)  # Red color for the rectangle
            text_color = (0, 0, 255)  # Red text
            label = "RED"
            confidence = min(area / 500.0, 1.0) * (aspect_ratio if aspect_ratio < 3 else 3/aspect_ratio)
        else:
            continue
        
        valid_traffic_lights.append((x, y, w, h, cx, cy, label, color, text_color, confidence))
    
    # Sort by confidence and take the top result
    if valid_traffic_lights:
        valid_traffic_lights.sort(key=lambda x: x[9], reverse=True)
        x, y, w, h, cx, cy, label, color, text_color, _ = valid_traffic_lights[0]
        
        detected_label = label
        
        # IMPORTANT: Adjust coordinates to the original frame by adding ROI offsets
        x_orig = x + roi_x_min
        y_orig = y + roi_y_min
        cx_orig = cx + roi_x_min
        cy_orig = cy + roi_y_min
        
        # Draw rectangle on the original frame with adjusted coordinates
        cv2.rectangle(frame, (x_orig, y_orig), (x_orig + w, y_orig + h), color, 2)
            
        # Draw the center point on the original frame
        cv2.circle(frame, (cx_orig, cy_orig), 3, (255, 0, 0), -1)
        
        # Display text on the original frame
        cv2.putText(frame, label, (x_orig, y_orig - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    
    # Make sure we always return a label, even if it's None
    # This ensures compatibility with pmain1.py
    return frame, detected_label