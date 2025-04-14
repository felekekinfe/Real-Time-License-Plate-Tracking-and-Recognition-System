import cv2

# Load image
img = cv2.imread('images.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the image
ret, thresh1 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

# Invert the binary image
inverted = cv2.bitwise_not(img)
ret, thresh2 = cv2.threshold(inverted, 100, 255, cv2.THRESH_BINARY_INV)
inverted= cv2.bilateralFilter(inverted, 9, 75, 75)




# Show everything
cv2.imshow("Original", img)
cv2.imshow("Threshold 1", thresh1)
cv2.imshow("Threshold 2", thresh2)
cv2.imshow("Inverted", inverted)

cv2.waitKey(0)
cv2.destroyAllWindows()
