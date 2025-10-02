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
