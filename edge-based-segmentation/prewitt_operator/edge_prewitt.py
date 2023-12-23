import cv2
import numpy as np
from matplotlib import pyplot as plt


# Load the image
img = cv2.imread('edge_prewitt_sample.png')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian smoothing
blur = cv2.GaussianBlur(gray, (3, 3), 0)

# Define the Prewitt kernel for the x and y directions
prewittx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewitty = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

# Calculate the Prewitt gradient in the x and y directions
sobelx = cv2.filter2D(blur, -1, prewittx)
sobely = cv2.filter2D(blur, -1, prewitty)

sobelx = sobelx.astype(np.float32)
sobely = sobely.astype(np.float32)

# Compute the gradient magnitude and direction
mag, angle = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)

# Threshold the magnitude to obtain the edges
edges = cv2.threshold(mag, 50, 255, cv2.THRESH_BINARY)[1]

# Display the result
plt.subplot(211),plt.imshow(img)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(212),plt.imshow(edges, 'gray')
plt.title("edge"), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()
cv2.imwrite("output.jpg",edges)