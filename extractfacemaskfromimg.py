import cv2
from cvzone.SelfiSegmentationModule import SelfiSegmentation
from PIL import Image
import os
import numpy as np
import sys
import pprint
import math
from cvzone.PoseModule import PoseDetector
import mediapipe as mp
segmentor = SelfiSegmentation(model=0)
pp = pprint.PrettyPrinter(indent=4)
np.set_printoptions(threshold=sys.maxsize)
detector = PoseDetector(staticMode=False,
                        modelComplexity=1,
                        smoothLandmarks=True,
                        enableSegmentation=False,
                        smoothSegmentation=True,
                        detectionCon=0.5,
                        trackCon=0.5)
                        
 
input_path = 'D:\\VTON\\overlay\\public4.jpg'

mpDraw = mp.solutions.drawing_utils 
mpPose = mp.solutions.pose  
pose = mpPose.Pose() 






img = cv2.imread(input_path) 

def slope_intercept(p1,p2):
# print(p1,p2)
    slope=(p2[1]-p1[1])/(p2[0]-p1[0])
    # print(math.degrees(math.atan(slope)))
    intercept=p1[1]-slope*p1[0]
    return slope,intercept

# print(img.shape)
def click_event(event, x, y, flags, params):
   global imgOut
   if event == cv2.EVENT_LBUTTONDOWN:
      print(f'({x},{y})')
      
      # put coordinates as text on the image
      cv2.putText(imgOut, f'({x},{y})',(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
      
      # draw point on the image
      cv2.circle(imgOut, (x,y), 3, (0,255,255), -1)
 
def xy_coordinate_positions(positions):
    real_positions = {}
    for key in positions:
        real_positions[key]=[positions[key][0],-1*positions[key][1]]
        
    return real_positions    


def img_position_from_xy_coordinate_positions(positions):
    img_positions = {}
    for key in positions:
        img_positions[key]=[positions[key][0],-1*positions[key][1]]
        
    return img_positions
    
def get_midpoint(p1,p2):
    midpoint=[round((p1[0]+p2[0])/2),round((p1[1]+p2[1])/2)]
    return midpoint
                        





imgOut = segmentor.removeBG(img, imgBg=(255, 255, 255), cutThreshold=0.4)


# results = pose.process(imgOut)
# pp.pprint("Results from process:")
# pp.pprint(vars(results))

pose=detector.findPose(imgOut,draw=False)
lmList, bboxInfo = detector.findPosition(pose,draw=False, bboxWithHands=False)
# pp.pprint("Results from lmList:")
# pp.pprint(lmList)
positions = {
"left_eye" : lmList[3][:2],
"right_eye": lmList[6][:2],
"nose" : lmList[0][:2],
"left_shoulder" : lmList[11][:2],
"right_shoulder" : lmList[12][:2]
}

pp.pprint(positions)




xy_coordinate_positions=xy_coordinate_positions(positions)
pp.pprint(xy_coordinate_positions)

eye_midpoint=get_midpoint(xy_coordinate_positions["left_eye"],xy_coordinate_positions["right_eye"])
thorax_midpoint=get_midpoint(xy_coordinate_positions["left_shoulder"],xy_coordinate_positions["right_shoulder"])
xy_coordinate_positions["eye_midpoint"]=eye_midpoint
xy_coordinate_positions["thorax_midpoint"]=thorax_midpoint

face_nose_thorax_distance=math.dist(xy_coordinate_positions["nose"],xy_coordinate_positions["thorax_midpoint"])
reduced_circle_radius=round(face_nose_thorax_distance * 40/100)
pp.pprint("reduced circle")
pp.pprint(reduced_circle_radius)
positions=img_position_from_xy_coordinate_positions(xy_coordinate_positions)
pp.pprint(positions)

# sys.exit()
for key in positions:
    cv2.circle(imgOut, (positions[key][0],positions[key][1]), radius=3, color=(0, 255, 0), thickness=-1)



cv2.circle(imgOut,positions["thorax_midpoint"],radius=reduced_circle_radius,color=(0,0,255),thickness=1)

cv2.imwrite("D:\\VTON\\overlay\\human_image9.jpg",imgOut)
cv2.namedWindow("Masked Image")
cv2.setMouseCallback("Masked Image", click_event)


# print(imgOut.shape)
while True:
   cv2.imshow("Masked Image", imgOut)
   if cv2.waitKey(10) & 0xFF == ord('q'):
      break

cv2.destroyAllWindows()


# (789,486)
# (767,578)
# (684,512)
# (880,555)