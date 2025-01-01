import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

def detect_crowd_density_video(video_path, heatmap_path, overlay_path, boundingbox_path):
    # Read video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create VideoWriter objects for each output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_heatmap = cv2.VideoWriter(heatmap_path, fourcc, fps, (frame_width, frame_height))
    out_overlay = cv2.VideoWriter(overlay_path, fourcc, fps, (frame_width, frame_height))
    out_boundingbox = cv2.VideoWriter(boundingbox_path, fourcc, fps, (frame_width, frame_height))
    
    # Set up blob detector
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
    
    detector = cv2.SimpleBlobDetector_create(params)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Convert to grayscale and enhance contrast
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Detect blobs
        keypoints = detector.detect(enhanced)
        detection_points = [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]
        
        # Generate heatmap
        heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
        
        for x, y in detection_points:
            y_coord, x_coord = np.ogrid[-30:31, -30:31]
            g = np.exp(-(x_coord * x_coord + y_coord * y_coord) / (2 * 15**2))
            
            y_min = max(0, y - 30)
            y_max = min(frame_height, y + 31)
            x_min = max(0, x - 30)
            x_max = min(frame_width, x + 31)
            
            g_height = y_max - y_min
            g_width = x_max - x_min
            if g_height > 0 and g_width > 0:
                g_cropped = g[:g_height, :g_width]
                heatmap[y_min:y_max, x_min:x_max] += g_cropped
        
        # Apply filters to heatmap
        heatmap = gaussian_filter(heatmap, sigma=5)
        heatmap = maximum_filter(heatmap, size=3)
        
        # Normalize and colorize heatmap
        heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        
        # Create overlay
        overlay = cv2.addWeighted(frame, 0.7, heatmap_color, 0.3, 0)
        
        # Create bounding box frame
        boundingbox_frame = frame.copy()
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            size = int(kp.size)
            x_min = max(0, x - size // 2)
            y_min = max(0, y - size // 2)
            x_max = min(frame_width, x + size // 2)
            y_max = min(frame_height, y + size // 2)
            cv2.rectangle(boundingbox_frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        
        # Add crowd count to bounding box frame
        crowd_count = len(detection_points)
        cv2.putText(boundingbox_frame, f"Detected People: {crowd_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Write frames to output videos
        out_heatmap.write(heatmap_color)
        out_overlay.write(overlay)
        out_boundingbox.write(boundingbox_frame)
    
    # Release everything
    cap.release()
    out_heatmap.release()
    out_overlay.release()
    out_boundingbox.release()
    
    print(f"Processing complete. {frame_count} frames processed.")

# Paths for input and output files
video_path = r"C:\Users\krish\OneDrive\Desktop\video.mp4"
heatmap_path = r"C:\Users\krish\OneDrive\Desktop\heatmap.mp4"
overlay_path = r"C:\Users\krish\OneDrive\Desktop\overlay.mp4"
boundingbox_path = r"C:\Users\krish\OneDrive\Desktop\boundingbox.mp4"

# Process the video
detect_crowd_density_video(video_path, heatmap_path, overlay_path, boundingbox_path)