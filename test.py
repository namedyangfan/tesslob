try:
    import Image
except ImportError:
    from PIL import Image
import pytesseract

import cv2
import numpy as np

kernel = np.ones((5,5),np.uint8)

image_name = 'hard2.jpg'
img = cv2.imread(image_name)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

detector = cv2.SimpleBlobDetector_create()
# Detect blobs.
keypoints = detector.detect(gray)
print (keypoints[0].pt)
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
# im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
# cv2.imshow("Keypoints", im_with_keypoints)
# cv2.waitKey(0)


# gray = cv2.GaussianBlur(gray,(5,5),0)

# th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
#   cv2.THRESH_BINARY,11,2)


# th1 = cv2.threshold(th3, 0, 255,
#     cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# cv2.imshow('th3',th3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite('gray.png', gray)

# text = pytesseract.image_to_string(Image.open('gray.png'), lang='eng', boxes=True, config="hocr")
# # text = text.encode('ascii', 'ignore').decode('ascii')
# text = text.encode('utf-8')
# print(text)

# with open("Output.txt", "w") as fhandl:
#   fhandl.write(text)
