import cv2
import numpy as np

def lane_detection(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and help with edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Define a region of interest (ROI)
    height, width = edges.shape
    mask = np.zeros_like(edges)
    region_of_interest_vertices = np.array([[(0, height), (width / 2, height / 2), (width, height)]], dtype=np.int32)
    cv2.fillPoly(mask, region_of_interest_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Apply Hough Transform to detect lines in the image
    lines = cv2.HoughLinesP(masked_edges, 2, np.pi / 180, 100, minLineLength=50, maxLineGap=50)

    # Draw the detected lines on the original image
    line_image = np.zeros_like(image)
    draw_lines(line_image, lines)

    # Combine the original image with the detected lines
    result = cv2.addWeighted(image, 0.8, line_image, 1, 0)

    return result

def draw_lines(image, lines, color=(244, 194, 194), thickness=5):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)

# Example usage with an image file
input_image = cv2.imread('road_lane.png')
output_image = lane_detection(input_image)

# Display the result
cv2.imshow('Lane Detection', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
