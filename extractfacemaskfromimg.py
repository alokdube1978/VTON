import cv2
from cvzone.SelfiSegmentationModule import SelfiSegmentation
from PIL import Image
import os
import numpy as np
import sys
import pprint
import math
import threading
from cvzone.PoseModule import PoseDetector
import mediapipe as mp
from mediapipe.python._framework_bindings import image
from mediapipe.python._framework_bindings import image_frame
from mediapipe.tasks.python import vision
from mediapipe import tasks

POSEDETECTOR_BODY_PARTS=["nose","left eye (inner)","left eye","left eye (outer)","right eye (inner)",
"right eye","right eye (outer)","left ear","right ear","mouth (left)","mouth (right)",
"left shoulder","right shoulder","left elbow","right elbow","left wrist","right wrist",
"left pinky","right pinky","left index","right index","left thumb","right thumb",
"left hip","right hip","left knee","right knee","left ankle","right ankle",
"left heel","right heel","left foot index","right foot index"]

lock = threading.Lock()  # a lock on global scope, or self.lock = threading.Lock() in your class's __init_


np.seterr(divide='ignore', invalid='ignore')
input_path = './overlay/public2.jpg'
output_path="./overlay/human_image6.jpg"
model_path="./Models/selfie_segmenter.tflite"
mp_model_path = './Models/pose_landmarker_heavy.task'


BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

#we also try pose landmarker directly from base model
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

with open(model_path, 'rb') as f:
    model_data = f.read()
base_options = BaseOptions(model_asset_buffer=model_data)
options = vision.ImageSegmenterOptions(base_options=base_options,running_mode=VisionRunningMode.IMAGE,
                                              output_category_mask=1)


with open(mp_model_path, 'rb') as mp_f:
    mp_model_data = mp_f.read()
mp_options = PoseLandmarkerOptions(
base_options=BaseOptions(model_asset_buffer=mp_model_data),running_mode=VisionRunningMode.IMAGE)

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
                        


def getSelfieImageandFaceLandMarkPoints(img,RUN_CV_SELFIE_SEGMENTER=True,use_different_horizontal_vertical_scale=False,force_shoulder_z_alignment=False):
    global lock
    global pose,detector,options,base_options,POSEDETECTOR_BODY_PARTS
    xy_coordinate_positions={}
    
    with lock:
        if (RUN_CV_SELFIE_SEGMENTER==True):

                rembg_image = Selfie_segmentor.removeBG(img, imgBg=BG_COLOR, cutThreshold=0.48)
                imgOut=rembg_image
        else:
                imgOut=img
                rembg_image=Selfie_segmentor.removeBG(img, imgBg=BG_COLOR, cutThreshold=0.48)
        # # cv2.imshow("Selfie Masked",imgOut)
        # #we run it once more through mediapipe selife segmentor
        
        # if (RUN_CV_SELFIE_SEGMENTER==False):
            # print("second Segmenter")
            # human_image_tf = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            # with vision.ImageSegmenter.create_from_options(options) as segmenter:
                # segmentation_result = segmenter.segment(human_image_tf)
            # image_data=human_image_tf.numpy_view().copy()
            # category_mask = segmentation_result.category_mask
            # bg_image = np.zeros(image_data.shape, dtype=np.uint8)
            # bg_image[:] = BG_COLOR
            # category_mask_condition=np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.9
            # imgOut = np.where(category_mask_condition,  bg_image,image_data)
        # cv2.imshow("BG Masked",imgOut)
    with lock:
        pose=detector.findPose(img,draw=False)
    
        
    
    lmList, bboxInfo = detector.findPosition(pose,draw=False, bboxWithHands=False)
    # pp.pprint("Results from lmList:")
    # pp.pprint(lmList)
    positions = {
    "left_eye" : lmList[3][:2],
    "right_eye": lmList[6][:2],
    "nose" : lmList[0][:2],
    "left_shoulder" : lmList[11][:2],
    "right_shoulder" : lmList[12][:2],
    "left_ear":lmList[7][:2],
    "right_ear":lmList[8][:2],
    }
    #we get normalized z depth of shoulders too for reference, we use original image img here
    #image without background is better able to detect z depth
    with PoseLandmarker.create_from_options(mp_options) as landmarker:
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        image_height, image_width, _ = rgb_image.shape
        pose_landmarker_result = landmarker.detect(mp_image)
        # print(pose_landmarker_result,file=sys.stderr, flush=True)
        mp_pose_landmark_list={}
        for index, elem in enumerate(POSEDETECTOR_BODY_PARTS):
                x=round(pose_landmarker_result.pose_landmarks[0][index].x*image_width)
                y=round(pose_landmarker_result.pose_landmarks[0][index].y*image_height)
                z=pose_landmarker_result.pose_landmarks[0][index].z
                # print(elem,":",pose_landmarker_result.pose_landmarks[0][index])
                mp_pose_landmark_list[elem]=[]
                mp_pose_landmark_list[elem]=[x,y,z]
        positions.update({
            "mp_left_right_shoulder_z_distance":mp_pose_landmark_list["left shoulder"][2]-mp_pose_landmark_list["right shoulder"][2],
            })
        
        
        # if we need to ensure shoulders are aligned in z plane - we return error if they are not aligned
        if (force_shoulder_z_alignment==True and (abs(positions["mp_left_right_shoulder_z_distance"])>0.2)):
            print("Error!! Z values of left and right shoulder points not aligned")
            raise Exception('force_shoulder_z_alignment')
        
    print("Positions",file=sys.stderr, flush=True)
    print(positions,file=sys.stderr, flush=True)
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
    shoulder_points_distance=math.dist(xy_coordinate_positions["left_shoulder"],xy_coordinate_positions["right_shoulder"])
    xy_coordinate_positions["shoulder_points_distance"]=shoulder_points_distance
    
    xy_coordinate_positions["face_nose_thorax_distance"]=face_nose_thorax_distance
    
    if use_different_horizontal_vertical_scale==True:
        
        xy_coordinate_positions["horizontal_reduced_circle_radius"]=round(xy_coordinate_positions["shoulder_points_distance"]/2* 40/100)
        xy_coordinate_positions["vertical_reduced_circle_radius"]=round(xy_coordinate_positions["face_nose_thorax_distance"] * 40/100)
    else:
        xy_coordinate_positions["vertical_reduced_circle_radius"]=round(xy_coordinate_positions["face_nose_thorax_distance"] * 40/100)
        xy_coordinate_positions["horizontal_reduced_circle_radius"]=xy_coordinate_positions["vertical_reduced_circle_radius"]
    
    # if we are too wide on the vertical scale or horizontal scale- we tie it to horizontal scale
    if ((xy_coordinate_positions["vertical_reduced_circle_radius"]/xy_coordinate_positions["horizontal_reduced_circle_radius"])>1.23
        or (xy_coordinate_positions["horizontal_reduced_circle_radius"]/xy_coordinate_positions["vertical_reduced_circle_radius"])>1.23
        ):
        xy_coordinate_positions["vertical_reduced_circle_radius"]=round(xy_coordinate_positions["face_nose_thorax_distance"] * 40/100)
        xy_coordinate_positions["horizontal_reduced_circle_radius"]=xy_coordinate_positions["vertical_reduced_circle_radius"]
    
    vertical_reduced_circle_radius=xy_coordinate_positions["vertical_reduced_circle_radius"]
    horizontal_reduced_circle_radius=xy_coordinate_positions["horizontal_reduced_circle_radius"]
    
    xy_coordinate_positions["reduced_circle_radius"]=round(xy_coordinate_positions["face_nose_thorax_distance"] * 40/100)
    reduced_circle_radius=xy_coordinate_positions["reduced_circle_radius"]
    xy_coordinate_positions["thorax_top_bottom_distance"]=xy_coordinate_positions["horizontal_reduced_circle_radius"]*2
    nose_slope,nose_intercept=slope_intercept(xy_coordinate_positions["nose"],xy_coordinate_positions["thorax_midpoint"])
    shoulder_slope,shoulder_intercept=slope_intercept(xy_coordinate_positions["left_shoulder"],xy_coordinate_positions["right_shoulder"])
    xy_coordinate_positions["nose_slope"]=nose_slope
    xy_coordinate_positions["shoulder_slope"]=shoulder_slope
    
    #we align with nose thorax slope if it is off upto 10 degrees rather than shoulder slope 
    # if shoulder slope >3.5
    print ("Original Nose Slope",file=sys.stderr, flush=True)
    print(math.degrees(math.atan(xy_coordinate_positions["nose_slope"])),file=sys.stderr, flush=True)
    print ("Original Shoulder Slope",file=sys.stderr, flush=True)
    print(math.degrees(math.atan(xy_coordinate_positions["shoulder_slope"])),file=sys.stderr, flush=True)
    
        
    if ((abs(math.degrees(math.atan(nose_slope)))>=83 and abs(math.degrees(math.atan(nose_slope))) <=97 ) and (abs(math.degrees(math.atan(shoulder_slope)))>3.5)) :
       print ("Resetting shoulder slope as nose slope is vertical",file=sys.stderr, flush=True)
       if ((math.atan(nose_slope)>=0) and (math.atan(nose_slope)<=90)):
            shoulder_slope=abs(math.atan(nose_slope))-math.pi/2
       else:
           shoulder_slope=math.pi/2-abs(math.atan(nose_slope))
       xy_coordinate_positions["shoulder_slope"]=shoulder_slope
       print("Reset Shoulder slope",file=sys.stderr, flush=True)
       print(math.degrees(math.atan(xy_coordinate_positions["shoulder_slope"])),file=sys.stderr, flush=True)
       
 
    
    
    # print("----shoulder slope,intercept----",file=sys.stderr, flush=True)
    # print (shoulder_slope,shoulder_intercept,file=sys.stderr, flush=True)
    
   
    # print(math.sin(math.atan(shoulder_slope)))
    # if (nose_slope<=0):
        # xy_coordinate_positions["thorax_top"][0]=round(xy_coordinate_positions["thorax_midpoint"][0]+reduced_circle_radius*math.cos(math.pi+math.atan(nose_slope)))
        # xy_coordinate_positions["thorax_top"][1]=round(xy_coordinate_positions["thorax_midpoint"][1]+reduced_circle_radius*math.sin(math.pi+math.atan(nose_slope)))
    # else:
        # xy_coordinate_positions["thorax_top"][0]=round(xy_coordinate_positions["thorax_midpoint"][0]+reduced_circle_radius*math.cos(math.atan(nose_slope)))
        # xy_coordinate_positions["thorax_top"][1]=round(xy_coordinate_positions["thorax_midpoint"][1]+reduced_circle_radius*math.sin(math.atan(nose_slope)))

    # if (nose_slope<=0):
        # xy_coordinate_positions["thorax_bottom"][0]=round(xy_coordinate_positions["thorax_midpoint"][0]+reduced_circle_radius*math.cos(math.atan(nose_slope)))
        # xy_coordinate_positions["thorax_bottom"][1]=round(xy_coordinate_positions["thorax_midpoint"][1]+reduced_circle_radius*math.sin(math.atan(nose_slope)))
    # else:
        # xy_coordinate_positions["thorax_bottom"][0]=round(xy_coordinate_positions["thorax_midpoint"][0]+reduced_circle_radius*math.cos(math.pi+math.atan(nose_slope)))
        # xy_coordinate_positions["thorax_bottom"][1]=round(xy_coordinate_positions["thorax_midpoint"][1]+reduced_circle_radius*math.sin(math.pi+math.atan(nose_slope)))    


    if (shoulder_slope<=0):
        xy_coordinate_positions["right_shoulder_pivot"][0]=round(xy_coordinate_positions["thorax_midpoint"][0]+horizontal_reduced_circle_radius*math.cos(math.pi+math.atan(shoulder_slope)))
        xy_coordinate_positions["right_shoulder_pivot"][1]=round(xy_coordinate_positions["thorax_midpoint"][1]+horizontal_reduced_circle_radius*math.sin(math.pi+math.atan(shoulder_slope)))
    else:
        xy_coordinate_positions["right_shoulder_pivot"][0]=round(xy_coordinate_positions["thorax_midpoint"][0]+horizontal_reduced_circle_radius*math.cos(math.pi+math.atan(shoulder_slope)))
        xy_coordinate_positions["right_shoulder_pivot"][1]=round(xy_coordinate_positions["thorax_midpoint"][1]+horizontal_reduced_circle_radius*math.sin(math.pi+math.atan(shoulder_slope)))   



    if (shoulder_slope<=0):
        xy_coordinate_positions["left_shoulder_pivot"][0]=round(xy_coordinate_positions["thorax_midpoint"][0]+horizontal_reduced_circle_radius*math.cos(math.atan(shoulder_slope)))
        xy_coordinate_positions["left_shoulder_pivot"][1]=round(xy_coordinate_positions["thorax_midpoint"][1]+horizontal_reduced_circle_radius*math.sin(math.atan(shoulder_slope)))
    else:
        xy_coordinate_positions["left_shoulder_pivot"][0]=round(xy_coordinate_positions["thorax_midpoint"][0]+horizontal_reduced_circle_radius*math.cos(math.atan(shoulder_slope)))
        xy_coordinate_positions["left_shoulder_pivot"][1]=round(xy_coordinate_positions["thorax_midpoint"][1]+horizontal_reduced_circle_radius*math.sin(math.atan(shoulder_slope)))




    xy_coordinate_positions["thorax_top"][0]=round(xy_coordinate_positions["thorax_midpoint"][0]+vertical_reduced_circle_radius*math.cos(math.pi/2+math.atan(shoulder_slope)))
    xy_coordinate_positions["thorax_top"][1]=round(xy_coordinate_positions["thorax_midpoint"][1]+vertical_reduced_circle_radius*math.sin(math.pi/2+math.atan(shoulder_slope)))
    
    xy_coordinate_positions["thorax_bottom"][0]=round(xy_coordinate_positions["thorax_midpoint"][0]+vertical_reduced_circle_radius*math.cos(-1*math.pi/2+math.atan(shoulder_slope)))
    xy_coordinate_positions["thorax_bottom"][1]=round(xy_coordinate_positions["thorax_midpoint"][1]+vertical_reduced_circle_radius*math.sin(-1*math.pi/2+math.atan(shoulder_slope)))
    

    # print("reduced circle")
    # print(xy_coordinate_positions["reduced_circle_radius"])
    positions=img_position_from_xy_coordinate_positions(xy_coordinate_positions)
    print ("XY coordinate positions",file=sys.stderr, flush=True)
    print (xy_coordinate_positions,file=sys.stderr, flush=True)
    print ("Image postions",file=sys.stderr, flush=True)
    print(positions,file=sys.stderr, flush=True)
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
