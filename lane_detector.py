import cv2
import numpy as np
import matplotlib.pyplot as plt

def lane_detection(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    
    lane_image = np.copy(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 40, 120)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 70, 70])
    upper_yellow = np.array([50, 255, 255])
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 40, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    color_mask = cv2.bitwise_or(yellow_mask, white_mask)
    enhanced_edges = cv2.bitwise_or(edges, color_mask)

    def region_of_interest(img):
        height, width = img.shape
        polygon = np.array([
            [(int(width*0.02), height), 
             (int(width*0.98), height), 
             (int(width*0.60), int(height*0.6)), 
             (int(width*0.40), int(height*0.6))]
        ], dtype=np.int32)
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, polygon, 255)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    masked_edges = region_of_interest(enhanced_edges)

    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi/180,
        threshold=30,
        minLineLength=15,
        maxLineGap=250
    )
    
    if lines is None or len(lines) == 0:
        return image, masked_edges

    height, width, _ = image.shape
    left_lines = []
    right_lines = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length < 25:
                continue
            if abs(slope) < 0.15 or abs(slope) > 3.5:
                continue
            mid_x = (x1 + x2) / 2
            if slope < 0 and mid_x < width/2:
                left_lines.append(line)
            elif slope > 0 and mid_x > width/2:
                right_lines.append(line)

    lane_lines_image = np.zeros_like(image)

    def average_lane_lines(lines, default_slope=None):
        if not lines:
            return None
        slopes = []
        intercepts = []
        lengths = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                slopes.append(slope)
                intercepts.append(intercept)
                lengths.append(length)
        if slopes:
            total_length = sum(lengths)
            avg_slope = sum(s * l for s, l in zip(slopes, lengths)) / total_length
            avg_intercept = sum(i * l for i, l in zip(intercepts, lengths)) / total_length
            return avg_slope, avg_intercept
        elif default_slope:
            center_x = width / 2
            center_y = height * 0.7
            intercept = center_y - default_slope * center_x
            return default_slope, intercept
        return None

    def get_line_coordinates(y1, y2, slope, intercept):
        if slope is None or intercept is None:
            return None
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        return [(x1, y1), (x2, y2)]

    y_bottom = height
    y_top = int(height * 0.6)
    left_avg = average_lane_lines(left_lines, default_slope=-0.7)
    left_line_coords = None
    if left_avg:
        left_slope, left_intercept = left_avg
        left_line_coords = get_line_coordinates(y_bottom, y_top, left_slope, left_intercept)
    right_avg = average_lane_lines(right_lines, default_slope=0.7)
    right_line_coords = None
    if right_avg:
        right_slope, right_intercept = right_avg
        right_line_coords = get_line_coordinates(y_bottom, y_top, right_slope, right_intercept)

    if left_line_coords:
        cv2.line(
            lane_lines_image,
            left_line_coords[0],
            left_line_coords[1],
            [0, 0, 255],
            20
        )
    if right_line_coords:
        cv2.line(
            lane_lines_image,
            right_line_coords[0],
            right_line_coords[1],
            [0, 0, 255],
            20
        )

    if left_line_coords and right_line_coords:
        lane_mask = np.zeros_like(image)
        pts = np.array([
            [left_line_coords[0], 
             left_line_coords[1], 
             right_line_coords[1], 
             right_line_coords[0]]
        ], dtype=np.int32)
        cv2.fillPoly(lane_mask, pts, [0, 200, 0])
        lane_lines_image = cv2.addWeighted(lane_lines_image, 1, lane_mask, 0.3, 0)

    cv2.imwrite('lane_lines_only.jpg', lane_lines_image)
    result_image = cv2.addWeighted(lane_image, 0.7, lane_lines_image, 1.5, 0)
    cv2.imwrite('final_lane_detection.jpg', result_image)
    return result_image, masked_edges

def display_results(original_image, processed_image, edges_image):
    cv2.imwrite('original_image.jpg', original_image)
    cv2.imwrite('edges_image.jpg', edges_image)
    cv2.imwrite('lane_detection_result.jpg', processed_image)
    try:
        plt.figure(figsize=(18, 12))
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        plt.subplot(131)
        plt.title('Original Image', fontsize=16)
        plt.imshow(original_rgb)
        plt.axis('off')
        plt.subplot(132)
        plt.title('Edge Detection', fontsize=16)
        plt.imshow(edges_image, cmap='gray')
        plt.axis('off')
        plt.subplot(133)
        plt.title('Lane Detection Result', fontsize=16)
        plt.imshow(processed_rgb)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('combined_results.jpg', dpi=300)
        plt.show()
    except Exception as e:
        print(f"Warning: Could not display results using matplotlib: {e}")
        print("However, all results have been saved as image files.")
    print("\nAll results have been saved as separate image files.")

def main():
    image_path = input("Enter the path to your road image: ")
    image_path = image_path.strip('"\'')
    try:
        print(f"Processing image: {image_path}")
        result_image, edges = lane_detection(image_path)
        original = cv2.imread(image_path)
        if original is None:
            original = np.copy(result_image)
        print("Processing complete. Displaying and saving results...")
        display_results(original, result_image, edges)
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found. Please check the path and try again.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
