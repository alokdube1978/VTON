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
from mediapipe.python._framework_bindings import image
from mediapipe.python._framework_bindings import image_frame
from mediapipe.tasks.python import vision
from mediapipe import tasks
np.seterr(divide='ignore', invalid='ignore')
input_path = './overlay/public2.jpg'
output_path="./overlay/human_image6.jpg"
model_path="./Models/selfie_multiclass_256x256.tflite"



BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

with open(model_path, 'rb') as f:
    model_data = f.read()
base_options = BaseOptions(model_asset_buffer=model_data)
options = vision.ImageSegmenterOptions(base_options=base_options,running_mode=VisionRunningMode.IMAGE,
                                              output_category_mask=1)


Selfie_segmentor = SelfiSegmentation(model=0)
pp = pprint.PrettyPrinter(indent=4)
np.set_printoptions(threshold=sys.maxsize)
detector = PoseDetector(staticMode=True,
                        modelComplexity=1,
                        smoothLandmarks=True,
                        enableSegmentation=False,
                        smoothSegmentation=True,
                        detectionCon=0.5,
                        trackCon=0.5)
                        
 

mpDraw = mp.solutions.drawing_utils 
mpPose = mp.solutions.pose  
pose = mpPose.Pose() 


BG_COLOR = (255, 255, 255) # white


def slope_intercept(p1,p2):
# print(p1,p2)
    slope=np.float64(p2[1]-p1[1])/(p2[0]-p1[0])
    # print(math.degrees(math.atan(slope)))
    intercept=p1[1]-slope*p1[0]
    return slope,intercept

# print(img.shape)
# def click_event(event, x, y, flags, params):
   # global imgOut
   # if event == cv2.EVENT_LBUTTONDOWN:
      # print(f'({x},{y})')
      
      # # put coordinates as text on the image
      # cv2.putText(imgOut, f'({x},{y})',(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
      
      # # draw point on the image
      # cv2.circle(imgOut, (x,y), 3, (0,255,255), -1)
 
def get_xy_coordinate_positions(positions):
    real_positions = {}
    for key in positions:
        if isinstance(positions[key], list):
           real_positions[key]=[positions[key][0],-1*positions[key][1]]
        else:
            real_positions[key]=positions[key]
        
    return real_positions    


def img_position_from_xy_coordinate_positions(positions):
    img_positions = {}
    for key in positions:
        if isinstance(positions[key], list):
            img_positions[key]=[positions[key][0],-1*positions[key][1]]
        else:
            img_positions[key]=positions[key]
        
    return img_positions
    
def get_midpoint(p1,p2):
    midpoint=[round((p1[0]+p2[0])/2),round((p1[1]+p2[1])/2)]
    return midpoint
                        


def getSelfieImageandFaceLandMarkPoints(img,RUN_CV_SELFIE_SEGMENTER=True):
    global pose,detector,options,base_options
    xy_coordinate_positions={}
    if (RUN_CV_SELFIE_SEGMENTER==True):
        imgOut = Selfie_segmentor.removeBG(img, imgBg=BG_COLOR, cutThreshold=0.48)
    # cv2.imshow("Selfie Masked",imgOut)
    #we run it once more through mediapipe selife segmentor
    
    if (RUN_CV_SELFIE_SEGMENTER==False):
        print("second Segmenter")
        human_image_tf = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        with vision.ImageSegmenter.create_from_options(options) as segmenter:
            segmentation_result = segmenter.segment(human_image_tf)
        image_data=human_image_tf.numpy_view().copy()
        category_mask = segmentation_result.category_mask
        bg_image = np.zeros(image_data.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        category_mask_condition=np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.9
        imgOut = np.where(category_mask_condition, image_data, bg_image)
    # cv2.imshow("BG Masked",imgOut)
    
    
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
    # print("Positions",file=sys.stderr, flush=True)
    # print(positions,file=sys.stderr, flush=True)
    xy_coordinate_positions=get_xy_coordinate_positions(positions)
    xy_coordinate_positions["thorax_top"]=[0,0]
    xy_coordinate_positions["thorax_bottom"]=[0,0]
    xy_coordinate_positions["right_shoulder_pivot"]=[0,0]
    xy_coordinate_positions["left_shoulder_pivot"]=[0,0]
    # print("xy_coordinate_positions")
    # print(xy_coordinate_positions,file=sys.stderr, flush=True)
    eye_midpoint=get_midpoint(xy_coordinate_positions["left_eye"],xy_coordinate_positions["right_eye"])
    thorax_midpoint=get_midpoint(xy_coordinate_positions["left_shoulder"],xy_coordinate_positions["right_shoulder"])
    xy_coordinate_positions["eye_midpoint"]=eye_midpoint
    xy_coordinate_positions["thorax_midpoint"]=thorax_midpoint

    face_nose_thorax_distance=math.dist(xy_coordinate_positions["nose"],xy_coordinate_positions["thorax_midpoint"])
    
    xy_coordinate_positions["face_nose_thorax_distance"]=face_nose_thorax_distance
    xy_coordinate_positions["reduced_circle_radius"]=round(xy_coordinate_positions["face_nose_thorax_distance"] * 40/100)
    reduced_circle_radius=xy_coordinate_positions["reduced_circle_radius"]
    xy_coordinate_positions["thorax_top_bottom_distance"]=xy_coordinate_positions["reduced_circle_radius"]*2
    nose_slope,nose_intercept=slope_intercept(xy_coordinate_positions["nose"],xy_coordinate_positions["thorax_midpoint"])
    shoulder_slope,shoulder_intercept=slope_intercept(xy_coordinate_positions["left_shoulder"],xy_coordinate_positions["right_shoulder"])
    # print("----shoulder slope,intercept----",file=sys.stderr, flush=True)
    # print (shoulder_slope,shoulder_intercept,file=sys.stderr, flush=True)
    
   
    # print(math.sin(math.atan(shoulder_slope)))
    if (nose_slope>=0):
        xy_coordinate_positions["thorax_top"][0]=round(xy_coordinate_positions["thorax_midpoint"][0]+reduced_circle_radius*math.cos(math.atan(nose_slope)))
        xy_coordinate_positions["thorax_top"][1]=round(xy_coordinate_positions["thorax_midpoint"][1]+reduced_circle_radius*math.sin(math.atan(nose_slope)))
    else:
        xy_coordinate_positions["thorax_top"][0]=round(xy_coordinate_positions["thorax_midpoint"][0]+reduced_circle_radius*math.cos(math.pi+math.atan(nose_slope)))
        xy_coordinate_positions["thorax_top"][1]=round(xy_coordinate_positions["thorax_midpoint"][1]+reduced_circle_radius*math.sin(math.pi+math.atan(nose_slope)))

    if (nose_slope>=0):
        xy_coordinate_positions["thorax_bottom"][0]=round(xy_coordinate_positions["thorax_midpoint"][0]+reduced_circle_radius*math.cos(math.pi+math.atan(nose_slope)))
        xy_coordinate_positions["thorax_bottom"][1]=round(xy_coordinate_positions["thorax_midpoint"][1]+reduced_circle_radius*math.sin(math.pi+math.atan(nose_slope)))
    else:
        xy_coordinate_positions["thorax_bottom"][0]=round(xy_coordinate_positions["thorax_midpoint"][0]+reduced_circle_radius*math.cos(math.atan(nose_slope)))
        xy_coordinate_positions["thorax_bottom"][1]=round(xy_coordinate_positions["thorax_midpoint"][1]+reduced_circle_radius*math.sin(math.atan(nose_slope)))    


    if (shoulder_slope<=0):
        xy_coordinate_positions["right_shoulder_pivot"][0]=round(xy_coordinate_positions["thorax_midpoint"][0]+reduced_circle_radius*math.cos(math.pi+math.atan(shoulder_slope)))
        xy_coordinate_positions["right_shoulder_pivot"][1]=round(xy_coordinate_positions["thorax_midpoint"][1]+reduced_circle_radius*math.sin(math.pi+math.atan(shoulder_slope)))
    else:
        xy_coordinate_positions["right_shoulder_pivot"][0]=round(xy_coordinate_positions["thorax_midpoint"][0]+reduced_circle_radius*math.cos(math.pi+math.atan(shoulder_slope)))
        xy_coordinate_positions["right_shoulder_pivot"][1]=round(xy_coordinate_positions["thorax_midpoint"][1]+reduced_circle_radius*math.sin(math.pi+math.atan(shoulder_slope)))   



    if (shoulder_slope<=0):
        xy_coordinate_positions["left_shoulder_pivot"][0]=round(xy_coordinate_positions["thorax_midpoint"][0]+reduced_circle_radius*math.cos(math.atan(shoulder_slope)))
        xy_coordinate_positions["left_shoulder_pivot"][1]=round(xy_coordinate_positions["thorax_midpoint"][1]+reduced_circle_radius*math.sin(math.atan(shoulder_slope)))
    else:
        xy_coordinate_positions["left_shoulder_pivot"][0]=round(xy_coordinate_positions["thorax_midpoint"][0]+reduced_circle_radius*math.cos(math.atan(shoulder_slope)))
        xy_coordinate_positions["left_shoulder_pivot"][1]=round(xy_coordinate_positions["thorax_midpoint"][1]+reduced_circle_radius*math.sin(math.atan(shoulder_slope)))
      

    # print("reduced circle")
    # print(xy_coordinate_positions["reduced_circle_radius"])
    positions=img_position_from_xy_coordinate_positions(xy_coordinate_positions)
    # print ("Image postions",file=sys.stderr, flush=True)
    # print(positions,file=sys.stderr, flush=True)
    return imgOut,positions


def draw_points_on_image(imgOut,positions):
    for key in positions:
        if isinstance(positions[key], list):
            cv2.circle(imgOut, (positions[key][0],positions[key][1]), radius=3, color=(0, 255, 0), thickness=-1)
    cv2.circle(imgOut,positions["thorax_midpoint"],radius=positions["reduced_circle_radius"],color=(0,0,255),thickness=1)
    return imgOut

def main():
    img = cv2.imread(input_path,cv2.IMREAD_UNCHANGED) 
    imgOut,positions=getSelfieImageandFaceLandMarkPoints(img,RUN_CV_SELFIE_SEGMENTER=True)
    imgOut=draw_points_on_image(imgOut,positions)
    cv2.namedWindow("Masked Image")
    # cv2.setMouseCallback("Masked Image", click_event)
    cv2.imshow("Masked Image", imgOut)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
