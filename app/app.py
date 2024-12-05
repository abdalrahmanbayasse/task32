import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# Function to apply Roberts edge detection manually
def roberts_edge_detection(image):
    # Define the Roberts Cross operator kernels
    kernel_x = np.array([[1, 0], [0, -1]])
    kernel_y = np.array([[0, 1], [-1, 0]])
    
    # Convolve the image with the kernels
    gradient_x = ndimage.convolve(image, kernel_x)
    gradient_y = ndimage.convolve(image, kernel_y)
    
    # Calculate the gradient magnitude
    roberts = np.hypot(gradient_x, gradient_y)
    
    return roberts

# Function to apply Prewitt edge detection
def prewitt_edge_detection(image):
    # Prewitt operator kernels for detecting horizontal and vertical edges
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    
    # Apply convolution using kernels
    edges_x = cv2.filter2D(image, -1, kernel_x)
    edges_y = cv2.filter2D(image, -1, kernel_y)
    
    # Combine horizontal and vertical edges
    prewitt = np.hypot(edges_x, edges_y)
    return prewitt

# Function to apply Sobel edge detection
def sobel_edge_detection(image):
    # Sobel operator for detecting edges in both x and y directions
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Combine gradients
    sobel = cv2.magnitude(sobel_x, sobel_y)
    return sobel

# Function to apply Canny edge detection
def canny_edge_detection(image):
    # Apply Canny edge detector
    canny = cv2.Canny(image, 100, 200)
    return canny

# Main function to load image, apply edge detection, and display results
def apply_edge_detection(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Image not found.")
        return

    # Apply edge detection methods
    roberts = roberts_edge_detection(image)
    prewitt = prewitt_edge_detection(image)
    sobel = sobel_edge_detection(image)
    canny = canny_edge_detection(image)

    # Display the results using Matplotlib
    plt.figure(figsize=(10, 8))

    # Plot Roberts edge detection result
    plt.subplot(2, 2, 1)
    plt.imshow(roberts, cmap='gray')
    plt.title('Roberts Edge Detection')
    plt.axis('off')

    # Plot Prewitt edge detection result
    plt.subplot(2, 2, 2)
    plt.imshow(prewitt, cmap='gray')
    plt.title('Prewitt Edge Detection')
    plt.axis('off')

    # Plot Sobel edge detection result
    plt.subplot(2, 2, 3)
    plt.imshow(sobel, cmap='gray')
    plt.title('Sobel Edge Detection')
    plt.axis('off')

    # Plot Canny edge detection result
    plt.subplot(2, 2, 4)
    plt.imshow(canny, cmap='gray')
    plt.title('Canny Edge Detection')
    plt.axis('off')

    # Show the plot with all the edge detection results
    plt.tight_layout()
    plt.show()

# Run the edge detection on the 'photo.png' image
apply_edge_detection('photo.jpg')