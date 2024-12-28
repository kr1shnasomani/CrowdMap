# Import the required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, maximum_filter

def detect_crowd_density(image_path, heatmap_path, overlay_path, boundingbox_path):
    # Read and preprocess the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Blob detection parameters
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10
    params.maxThreshold = 200
    params.filterByArea = True
    params.minArea = 50
    params.maxArea = 800
    params.filterByCircularity = False
    params.filterByConvexity = True
    params.minConvexity = 0.5
    params.filterByInertia = True
    params.minInertiaRatio = 0.1
    
    # Create blob detector
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(enhanced)
    
    # Get detection points
    detection_points = [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]
    
    if not detection_points:
        print("No crowd detected.")
        return None
    
    # Generate heatmap
    height, width = image.shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    for x, y in detection_points:
        y_coord, x_coord = np.ogrid[-30:31, -30:31]
        g = np.exp(-(x_coord * x_coord + y_coord * y_coord) / (2 * 15**2))
        
        y_min = max(0, y - 30)
        y_max = min(height, y + 31)
        x_min = max(0, x - 30)
        x_max = min(width, x + 31)
        
        g_height = y_max - y_min
        g_width = x_max - x_min
        if g_height > 0 and g_width > 0:
            g_cropped = g[:g_height, :g_width]
            heatmap[y_min:y_max, x_min:x_max] += g_cropped
    
    heatmap = gaussian_filter(heatmap, sigma=5)
    heatmap = maximum_filter(heatmap, size=3)
    
    # Normalize and colorize heatmap
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    cv2.imwrite(heatmap_path, heatmap_color)
    
    # Create overlay image (without detected count)
    overlay = cv2.addWeighted(image, 0.7, heatmap_color, 0.3, 0)
    cv2.imwrite(overlay_path, overlay)
    
    # Create bounding box image
    boundingbox_image = image.copy()
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        size = int(kp.size)
        x_min = max(0, x - size // 2)
        y_min = max(0, y - size // 2)
        x_max = min(width, x + size // 2)
        y_max = min(height, y + size // 2)
        cv2.rectangle(boundingbox_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
    
    # Add crowd count to bounding box image
    crowd_count = len(detection_points)
    cv2.putText(boundingbox_image, f"Detected People: {crowd_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 255), 2)
    cv2.imwrite(boundingbox_path, boundingbox_image)
    
    # Display results
    plt.figure(figsize=(15, 5))
    plt.subplot(141)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(142)
    plt.imshow(cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB))
    plt.title('Heatmap')
    plt.axis('off')
    
    plt.subplot(143)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title('Overlay')
    plt.axis('off')
    
    plt.subplot(144)
    plt.imshow(cv2.cvtColor(boundingbox_image, cv2.COLOR_BGR2RGB))
    plt.title('Bounding Boxes')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Detection complete. Crowd count: {crowd_count}")

# Paths for input and output files
image_path = r"C:\Users\krish\OneDrive\Desktop\image.jpg"
heatmap_path = r"C:\Users\krish\OneDrive\Desktop\heatmap.jpg"
overlay_path = r"C:\Users\krish\OneDrive\Desktop\overlay.jpg"
boundingbox_path = r"C:\Users\krish\OneDrive\Desktop\boundingbox.jpg"

detect_crowd_density(image_path, heatmap_path, overlay_path, boundingbox_path)