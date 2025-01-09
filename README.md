<h1 align="center">CrowdMap</h1>
The system analyzes images to detect people, estimate crowd density, generate heatmaps and display bounding boxes with crowd counts, using computer vision techniques for applications in safety, surveillance and urban planning.

## Execution Guide:
1. Run the following command line in the terminal:
   ```
   pip install opencv-python numpy matplotlib scipy
   ```

2. Enter the path of the image whose crowd density you wish to see

3. Enter the path of the folder where you want to save the bounding box, heatmap and overlay image

4. Upon running the code it displays the number of people detected along with 3 images (bounding box, heatmap and overlay)

## Model Prediction:

  Input: 
  
  ![image](https://github.com/user-attachments/assets/5841f800-759f-4b93-817c-44e5caa710d3)

  Output:

  a. `boundingbox.jpg`

  ![boundingbox](https://github.com/user-attachments/assets/e8e981c7-4e59-44dc-93f0-53e7b27a71e6)

  b. `heatmap.jpg`

  ![heatmap](https://github.com/user-attachments/assets/541ebabf-cef6-4ee6-8d9f-e72489648882)

  c. `overlay.jpg`

  ![overlay](https://github.com/user-attachments/assets/a4d7b2e2-f1aa-4d19-9898-ec1d77dbe0c3)
   
## Overview:
The project is an advanced solution for analyzing the crowd density in images. This project aims to detect people in crowded scenes, estimate crowd density, generate a heatmap representing the concentration of individuals and draw bounding boxes around detected people.

#### Key Features:
1. **Crowd Detection:** The system uses **computer vision techniques** to detect individuals in an image, highlighting their presence in the scene. It leverages the **blob detection** method for identifying and marking objects (people) in the image.
   
2. **Heatmap Generation:** After detecting individuals, the system generates a **heatmap** that visually represents the density of the crowd. Warmer areas of the map indicate regions with higher crowd density, while cooler areas indicate fewer people.
   
3. **Overlay Image:** The system overlays the detected crowd density onto the original image, combining the crowd detection and heatmap to visually indicate crowd concentration without modifying the original image.

4. **Bounding Box Visualization:** The system draws bounding boxes around detected individuals to clearly show where the people are located in the image. Each bounding box is labeled with the number of detected people.

5. **Crowd Count:** The system counts the number of detected people and displays this count on the bounding box image, providing an overview of the crowd size.

#### Technologies Used:
- **OpenCV** for image processing tasks such as blob detection, contrast enhancement, and bounding box drawing.
- **NumPy** for handling array operations and heatmap generation.
- **SciPy** for advanced image filtering and heatmap smoothing.
- **Matplotlib** for visualizing results, including the original image, heatmap, overlay, and bounding boxes.

The system is ideal for applications where real-time analysis of crowd size and density is crucial, such as in public events, transportation hubs, or urban monitoring.
