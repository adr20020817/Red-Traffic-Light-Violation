import os
import cv2
import mysql.connector
from paddleocr import PaddleOCR
from datetime import datetime
import glob
import numpy as np

# Initialize PaddleOCR once for efficiency
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

def create_database():
    """Create mysql database with required tables if they don't exist"""
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="violation_records"
        )
        cursor = conn.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS violation_records")
        cursor.execute("USE violation_records")
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS license_records (
            id INT AUTO_INCREMENT PRIMARY KEY,
            license_plate VARCHAR(50) NOT NULL,
            offense VARCHAR(100),
            timestamp DATETIME,
            image_path VARCHAR(255)
        )
        ''')
        conn.commit()
        print("Database connection established successfully")
        return conn, cursor
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        return None, None

def preprocess_license_plate_image(img):
    """
    Enhanced preprocessing for Tanzania license plates (two-line format)
    
    Args:
        img: Input image (numpy array)
    
    Returns:
        Preprocessed image optimized for two-line text detection
    """
    if img is None or img.size == 0:
        return None
    
    # Resize image for better OCR - make it larger for two-line plates
    height, width = img.shape[:2]
    if width < 300 or height < 100:
        # Calculate scale to make width at least 300px and height at least 100px
        scale_w = 300 / width if width < 300 else 1
        scale_h = 100 / height if height < 100 else 1
        scale = max(scale_w, scale_h)
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Apply bilateral filter to reduce noise while keeping edges sharp
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Sharpen the image to make text more distinct
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    gray = cv2.filter2D(gray, -1, kernel)
    
    # Convert back to BGR for PaddleOCR
    processed_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    return processed_img

def sort_text_lines_by_position(text_results):
    """
    Sort detected text lines by their vertical position (top to bottom)
    
    Args:
        text_results: PaddleOCR results containing bounding boxes and text
    
    Returns:
        Sorted list of text results from top to bottom
    """
    if not text_results:
        return []
    
    # Extract y-coordinates and sort by vertical position
    text_with_positions = []
    for result in text_results:
        if result is None:
            continue
        
        bbox = result[0]  # Bounding box coordinates
        text_info = result[1]  # Text and confidence
        
        # Calculate average y-coordinate of the bounding box
        y_coords = [point[1] for point in bbox]
        avg_y = sum(y_coords) / len(y_coords)
        
        text_with_positions.append((avg_y, result))
    
    # Sort by y-coordinate (top to bottom)
    text_with_positions.sort(key=lambda x: x[0])
    
    return [result for _, result in text_with_positions]

def clean_license_plate_text(text):
    """
    Clean and validate license plate text for Tanzania format
    
    Args:
        text: Raw OCR text
    
    Returns:
        Cleaned text or None if invalid
    """
    if not text:
        return None
    
    # Remove spaces and special characters, keep only alphanumeric
    cleaned = ''.join(c.upper() for c in text if c.isalnum())
    
    # Basic validation for license plate format
    if len(cleaned) < 1 or len(cleaned) > 15:  # More flexible for two-line plates
        return None
    
    return cleaned

def combine_tanzania_license_plate_lines(text_lines):
    """
    Combine multiple text lines into a proper Tanzania license plate format
    
    Args:
        text_lines: List of cleaned text lines from top to bottom
    
    Returns:
        Combined license plate text
    """
    if not text_lines:
        return None
    
    # Filter out very short or invalid lines
    valid_lines = [line for line in text_lines if line and len(line) >= 1]
    
    if not valid_lines:
        return None
    
    # For Tanzania plates, typically combine with a space or dash
    # You can modify this based on your specific format requirements
    if len(valid_lines) == 1:
        return valid_lines[0]
    elif len(valid_lines) == 2:
        # Two lines: combine them (e.g., "ABC" + "123" = "ABC 123")
        return f"{valid_lines[0]} {valid_lines[1]}"
    else:
        # More than two lines: take the two longest ones
        valid_lines.sort(key=len, reverse=True)
        return f"{valid_lines[0]} {valid_lines[1]}"

def detect_license_plate_from_image(image_path, offense="Red Light Violation"):
    """
    Detect and extract license plate text from Tanzania two-line license plates using PaddleOCR
    
    Args:
        image_path: Path to the image file
        offense: Type of offense (default is "Red Light Violation")
    
    Returns:
        Detected license plate text or None if not detected
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Error: Could not load image {image_path}")
            return None
        
        # Preprocess the image
        processed_img = preprocess_license_plate_image(img)
        if processed_img is None:
            print(f"Error: Image preprocessing failed for {image_path}")
            return None
        
        # Perform OCR using PaddleOCR with settings optimized for multi-line text
        results = ocr.ocr(processed_img, cls=True)
        
        if not results or not results[0]:
            print(f"No text detected in {os.path.basename(image_path)}")
            return None

        # Sort text results by vertical position (top to bottom)
        sorted_results = sort_text_lines_by_position(results[0])
        
        # Extract and clean text from each line
        text_lines = []
        total_confidence = 0
        valid_detections = 0
        
        print(f"Detected {len(sorted_results)} text line(s) in {os.path.basename(image_path)}:")
        
        for i, line in enumerate(sorted_results):
            if line is None:
                continue
                
            text_info = line[1]
            raw_text = text_info[0]
            confidence = text_info[1]
            
            print(f"  Line {i+1}: '{raw_text}' (confidence: {confidence:.3f})")
            
            # Clean the text
            cleaned_text = clean_license_plate_text(raw_text)
            
            if cleaned_text and confidence > 0.4:  # Lower threshold for multi-line
                text_lines.append(cleaned_text)
                total_confidence += confidence
                valid_detections += 1

        # Combine the text lines into final license plate
        if text_lines:
            combined_text = combine_tanzania_license_plate_lines(text_lines)
            avg_confidence = total_confidence / valid_detections if valid_detections > 0 else 0
            
            if combined_text:
                print(f"Final license plate: {combined_text} from {os.path.basename(image_path)} (avg confidence: {avg_confidence:.3f})")
                return combined_text

        print(f"No valid license plate detected in {os.path.basename(image_path)}")
        return None
            
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def insert_license_record(cursor, conn, license_plate, offense, image_path):
    """Insert license plate record into database"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO license_records (license_plate, offense, timestamp, image_path) VALUES (%s, %s, %s, %s)",
            (license_plate, offense, timestamp, image_path)
        )
        conn.commit()
        print(f"Added to database: {license_plate} with offense: {offense}")
        return True
    except mysql.connector.Error as err:
        print(f"Database insertion error: {err}")
        return False

def get_license_plates_folder_path(base_folder="license_plates", target_date=None):
    """
    Get the path to the license plates folder for a specific date
    
    Args:
        base_folder: Base folder name (default: "license_plates")
        target_date: Target date in YYYY-MM-DD format. If None, uses current date
    
    Returns:
        Full path to the date-specific folder
    """
    if target_date is None:
        target_date = datetime.now().strftime('%Y-%m-%d')
    
    folder_path = os.path.join(base_folder, target_date)
    return folder_path

def list_available_date_folders(base_folder="license_plates"):
    """
    List all available date folders in the license_plates directory
    
    Args:
        base_folder: Base folder name (default: "license_plates")
    
    Returns:
        List of available date folders
    """
    if not os.path.exists(base_folder):
        return []
    
    date_folders = []
    for item in os.listdir(base_folder):
        item_path = os.path.join(base_folder, item)
        if os.path.isdir(item_path):
            # Check if folder name matches date format (YYYY-MM-DD)
            try:
                datetime.strptime(item, '%Y-%m-%d')
                date_folders.append(item)
            except ValueError:
                continue
    
    return sorted(date_folders, reverse=True)  # Most recent first

def process_license_images_folder(base_folder="license_plates", target_date=None, offense="Red Light Violation"):
    """
    Process all images in the license_plates/date folder and add detected plates to database
    
    Args:
        base_folder: Base folder name (default: "license_plates")
        target_date: Target date in YYYY-MM-DD format. If None, uses current date
        offense: Type of offense for all processed images
    """
    # Create database connection
    conn, cursor = create_database()
    if conn is None or cursor is None:
        print("Failed to connect to database. Exiting.")
        return
    
    # Get the target folder path
    if target_date is None:
        target_date = datetime.now().strftime('%Y-%m-%d')
        folder_path = get_license_plates_folder_path(base_folder, target_date)
        print(f"Using current date: {target_date}")
    else:
        folder_path = get_license_plates_folder_path(base_folder, target_date)
        print(f"Using specified date: {target_date}")
    
    # Check if base folder exists
    if not os.path.exists(base_folder):
        print(f"Error: Base folder '{base_folder}' does not exist")
        conn.close()
        return
    
    # Check if date-specific folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist")
        
        # Show available date folders
        available_dates = list_available_date_folders(base_folder)
        if available_dates:
            print(f"\nAvailable date folders in '{base_folder}':")
            for date_folder in available_dates:
                print(f"  - {date_folder}")
            print(f"\nYou can specify a date using: process_license_images_folder(target_date='YYYY-MM-DD')")
        else:
            print(f"No date folders found in '{base_folder}'")
        
        conn.close()
        return
    
    # Supported image extensions
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    
    # Get all image files from the folder
    image_files = []
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, extension)))
        image_files.extend(glob.glob(os.path.join(folder_path, extension.upper())))
    
    if not image_files:
        print(f"No image files found in '{folder_path}' folder")
        conn.close()
        return
    
    print(f"Found {len(image_files)} image(s) in '{folder_path}' folder")
    print("Processing Tanzania license plate images...")
    print("-" * 60)
    
    processed_count = 0
    successful_detections = 0
    
    # Process each image
    for image_path in image_files:
        print(f"\nProcessing: {os.path.basename(image_path)}")
        
        # Detect license plate
        license_plate = detect_license_plate_from_image(image_path, offense)
        
        if license_plate:
            # Insert into database
            if insert_license_record(cursor, conn, license_plate, offense, image_path):
                successful_detections += 1
        
        processed_count += 1
    
    print("\n" + "=" * 60)
    print(f"Processing complete for date: {target_date}")
    print(f"Total images processed: {processed_count}")
    print(f"Successful detections: {successful_detections}")
    print(f"Failed detections: {processed_count - successful_detections}")
    
    # Display all records in the database
    display_database_records(cursor)
    
    # Close database connection
    conn.close()
    print("\nDatabase connection closed.")

def process_all_available_dates(base_folder="license_plates", offense="Red Light Violation"):
    """
    Process all available date folders in the license_plates directory
    
    Args:
        base_folder: Base folder name (default: "license_plates")
        offense: Type of offense for all processed images
    """
    available_dates = list_available_date_folders(base_folder)
    
    if not available_dates:
        print(f"No date folders found in '{base_folder}'")
        return
    
    print(f"Found {len(available_dates)} date folder(s) to process:")
    for date_folder in available_dates:
        print(f"  - {date_folder}")
    
    print("\nProcessing all date folders...")
    print("=" * 60)
    
    for date_folder in available_dates:
        print(f"\n>>> Processing date folder: {date_folder}")
        process_license_images_folder(base_folder, date_folder, offense)
        print(f">>> Completed processing: {date_folder}")

def display_database_records(cursor):
    """Display all records in the database"""
    try:
        cursor.execute("SELECT id, license_plate, offense, timestamp, image_path FROM license_records ORDER BY timestamp DESC")
        records = cursor.fetchall()
        
        print("\n" + "=" * 100)
        print("LICENSE PLATE DATABASE RECORDS:")
        print("=" * 100)
        print(f"{'ID':<5} {'License Plate':<15} {'Offense':<25} {'Timestamp':<20} {'Image Path':<35}")
        print("-" * 100)
        
        if records:
            for record in records:
                image_name = os.path.basename(record[4]) if record[4] else "N/A"
                print(f"{record[0]:<5} {record[1]:<15} {record[2]:<25} {str(record[3]):<20} {image_name:<35}")
        else:
            print("No records found in the database.")
            
    except mysql.connector.Error as err:
        print(f"Error retrieving records: {err}")

def main():
    """Main function to process license plate images"""
    print("License Plate OCR Processing System")
    print("=" * 50)
    
    # You can customize these parameters
    folder_path = "license_plates"  # Change this if your folder has a different name
    offense_type = "Red Light Violation"  # Change this if needed
    
    # Get current date for the folder path
    current_date = datetime.now().strftime('%Y-%m-%d')
    print(f"Current system date: {current_date}")
    
    # Option 1: Process current date folder (default behavior)
    print(f"\nProcessing license plates for current date: {current_date}")
    process_license_images_folder(folder_path, current_date, offense_type)
    
    # Option 2: Process a specific date (uncomment and modify as needed)
    # specific_date = "2025-01-15"  # Change to your desired date
    # print(f"\nProcessing specific date: {specific_date}")
    # process_license_images_folder(folder_path, specific_date, offense_type)
    
    # Option 3: Process all available date folders (uncomment if needed)
    # print("\nProcessing all available date folders...")
    # process_all_available_dates(folder_path, offense_type)

if __name__ == "__main__":
    main()