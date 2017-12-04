import pytesseract
import cv2, os
import numpy as np
from PIL import Image

class Reco:

  def __init__(self, dir, file_name):
    self.dir = dir
    self.image_path = os.path.join (dir, file_name)

  def load_image(self, ldebug=False):
    img = cv2.imread(self.image_path)
    self.gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    self.i_height, self.i_weith = self.gray.shape
    if ldebug: print( 'Dimention of image \nHeight:{} Width:{}'.format(self.i_height, self.i_weith))

  def set_blob_params(self, ldebug=False):
    self.params = cv2.SimpleBlobDetector_Params()
    self.params.minThreshold = 10;
    self.params.maxThreshold = 200;

    # Filter by Area.
    self.params.filterByArea = True
    self.params.minArea = 150

    # Filter by Circularity
    self.params.filterByCircularity = True
    self.params.minCircularity = 0.1
    # Filter by Inertia
    self.params.filterByInertia = True
    self.params.minInertiaRatio = 0.3
    # Filter by Convexity
    self.params.filterByConvexity = True
    self.params.minConvexity = 0.9

  def detect_blob(self, draw_match = False, ldebug=False):
    self.detector = cv2.SimpleBlobDetector_create(self.params)
    self.keypoints = self.detector.detect(self.gray)
    self.blob_x =[]
    self.blob_y = []

    for keypoint in self.keypoints:
      self.blob_x.append(keypoint.pt[0])
      self.blob_y.append(self.i_height - keypoint.pt[1])
    if ldebug: print(self.blob_y)

    if draw_match:
      # Draw detected blobs as red circles.
      # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
      im_with_keypoints = cv2.drawKeypoints(self.gray, self.keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
      # Show keypoints
      cv2.imshow("Keypoints", im_with_keypoints)
      cv2.waitKey(0)

  def tesseract(self, save_name = 'gray.png', ldebug=False):
    save_image_path = os.path.join(self.dir, save_name)
    cv2.imwrite(save_name, self.gray)
    img = Image.open(save_name)
    self.text_coord = pytesseract.image_to_string(img, lang=None, boxes=True, config="hocr")
    self.text = pytesseract.image_to_string(img, lang='eng')
    self.text = self.text.encode('ascii', 'ignore').decode('ascii')
    # self.text = self.text.encode('utf-8')

    self.text = [ i  for i in self.text.split('\n')]
    print(self.text)
    if ldebug: print(self.text_coord)

if __name__ == "__main__":
  dir = '/Users/fanyang/Project/OCR'
  file_name = 'hard3.jpg'
  test = Reco(dir, file_name)
  test.load_image(ldebug=True)
  test.set_blob_params()
  test.detect_blob(draw_match=True,ldebug=True)
  test.tesseract(ldebug=True)

