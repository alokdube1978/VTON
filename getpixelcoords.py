import cv2
import cvzone
import os
import math as math
import numpy as np
import sys
import pprint
# import PIL

# input_path = 'D:\\VTON\\overlay\\necklace-masked-img.png'

input_path = 'D:\\VTON\\overlay\\necklace9.png'

def back(*args):
    pass

def click_event(event, x, y, flags, params):
   if event == cv2.EVENT_LBUTTONDOWN:
      print(f'({x},{y})')
      
      # put coordinates as text on the image
      # cv2.putText(img, f'({x},{y})',(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
      
      # draw point on the image
      cv2.circle(img, (x,y), 30, (100,100,100), -1)


cv2.namedWindow("Image")
# cv2.createButton("Back",back,None,cv2.QT_PUSH_BUTTON,1)
cv2.setMouseCallback("Image", click_event)
img = cv2.imread(input_path,cv2.IMREAD_UNCHANGED)
while True:
   cv2.imshow("Image", img)
   if cv2.waitKey(10) & 0xFF == ord('q'):
      break
# cv2.imwrite(input_path,img)
cv2.destroyAllWindows()

# thorax bottom(365,476)
# thorax_top (365,273)
# right shoulder pivot(189,345)
# left shoulder pivot(540,345)