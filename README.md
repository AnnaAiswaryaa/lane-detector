# lane-detector
This project is a Lane Detection System built with Python, OpenCV, and NumPy.
It detects road lanes from input images by applying a pipeline of computer vision techniques and overlays detected lanes on the original road image.

The implementation supports:
Edge detection (Canny)
Color filtering (Yellow & White lane masks)
Region of Interest (ROI) masking
Lane line extraction using Hough Line Transform
Lane averaging for smooth detection
Highlighted output images

⚙️ Features

✅ Detects both left and right lanes
✅ Supports lane filling between detected lines
✅ Handles yellow and white lanes using HSV color filtering
✅ Saves processed outputs automatically:

 Tech Stack
Python 3.x
Libraries: OpenCV, NumPy, Matplotlib

 How It Works
Load road image from user input.
Convert image to grayscale & apply Gaussian blur.
Detect edges using Canny Edge Detection.
Filter yellow and white lane markings using HSV thresholds.
Apply ROI masking to focus only on the road area.
Use Hough Line Transform to detect line segments.
Average and extend lane lines for stability.

Overlay lane lines and shaded driving area on the original image.

Save all output images automatically.
