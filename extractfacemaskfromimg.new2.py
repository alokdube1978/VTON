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

interested_points=["left_shoulder","right_shoulder","nose","left_eye","right_eye","left_ear","right_ear"]
global_degrees_shoulder_slope_max=4
global_degrees_nose_slope_max=100
global_degrees_nose_slope_min=80
global_normalized_shoulders_z_limit=0.27
global_normalized_ears_z_limit=0.2
global_vertical_ratio=42
global_max_vertical_horizontal_ratio=1.15
global_max_horizontal_vertical_ratio=1.08
global_horizontal_ratio=50
global_shoulder_to_nose_eyes_ratio_max=9.2

global_vertical_offet=90-global_degrees_nose_slope_min
global_horizontal_offset=global_degrees_shoulder_slope_max

global_ear_to_eye_nose_ratio_min=1.5
global_ear_to_eye_nose_ratio_max=3.42

global_nose_thorax_to_nose_eyes_ratio_min=3.4
global_nose_thorax_to_nose_eyes_ratio_max=4.5
global_nose_thorax_to_nose_eyes_ratio_avg=4.2

POSEDETECTOR_BODY_PARTS=["nose","left eye (inner)","left eye","left eye (outer)","right eye (inner)",
"right eye","right eye (outer)","left ear","right ear","mouth (left)","mouth (right)",
"left shoulder","right shoulder","left elbow","right elbow","left wrist","right wrist",
"left pinky","right pinky","left index","right index","left thumb","right thumb",
"left hip","right hip","left knee","right knee","left ankle","right ankle",
"left heel","right heel","left foot index","right foot index"]

lock = threading.Lock()  
# a lock on global scope, or self.lock = threading.Lock() in your class's __init_


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
    if slope==-0.0:
        slope=0.0
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
def get_3d_midpoint(p1,p2):
    midpoint=[round((p1[0]+p2[0])/2),round((p1[1]+p2[1])/2),round((p1[2]+p2[2])/2)]
    return midpoint                       

def reset_thorax_midpoint(multiplier,eye_nose_distance,thorax_midpoint,nose_slope,nose):
    print ("Reseting Thorax Midpoint",file=sys.stderr, flush=True)
    print ("Old Thorax Midpoint:",file=sys.stderr, flush=True)
    print (thorax_midpoint,file=sys.stderr, flush=True)
    if (nose_slope>0):
        thorax_midpoint[0]=round(nose[0]-multiplier*eye_nose_distance*math.cos(math.atan(nose_slope)))
        thorax_midpoint[1]=round(nose[1]-multiplier*eye_nose_distance*math.sin(math.atan(nose_slope)))
    else:
        thorax_midpoint[0]=round(nose[0]+multiplier*eye_nose_distance*math.cos(math.atan(nose_slope)))
        thorax_midpoint[1]=round(nose[1]+multiplier*eye_nose_distance*math.sin(math.atan(nose_slope)))

    print ("New Thorax Midpoint:",file=sys.stderr, flush=True)
    print (thorax_midpoint,file=sys.stderr, flush=True)
    return thorax_midpoint
    

def getSelfieImageandFaceLandMarkPoints(img,RUN_CV_SELFIE_SEGMENTER=True,use_different_horizontal_vertical_scale=False,force_shoulder_z_alignment=False,use_cv_pose_detector=True,debug=True):
    global lock,global_shoulder_to_nose_eyes_ratio_max,global_nose_thorax_to_nose_eyes_ratio_avg,global_nose_thorax_to_nose_eyes_ratio_max,global_nose_thorax_to_nose_eyes_ratio_min
    global pose,detector,options,base_options,POSEDETECTOR_BODY_PARTS,global_horizontal_ratio, global_vertical_ratio
    global global_degrees_nose_slope_max, global_degrees_shoulder_slope_max,global_degrees_nose_slope_min,global_normalized_shoulders_z_limit,global_normalized_ears_z_limit
    xy_coordinate_positions={}
    positions={}
    global interested_points
    global global_max_vertical_horizontal_ratio, global_max_horizontal_vertical_ratio,global_vertical_offet, global_horizontal_offset
    degrees_shoulder_slope_max=global_degrees_shoulder_slope_max
    degrees_nose_slope_max=global_degrees_nose_slope_max
    degrees_nose_slope_min=global_degrees_nose_slope_min
    normalized_shoulders_z_limit=global_normalized_shoulders_z_limit
    normalized_ears_z_limit=global_normalized_ears_z_limit
    vertical_ratio=global_vertical_ratio
    
    vertical_offset=global_vertical_offet
    horizontal_offset=global_horizontal_offset
    
    max_vertical_horizontal_ratio=global_max_vertical_horizontal_ratio
    max_horizontal_vertical_ratio=global_max_horizontal_vertical_ratio
    horizontal_ratio=global_horizontal_ratio
    shoulder_to_nose_eyes_ratio_max=global_shoulder_to_nose_eyes_ratio_max

    nose_thorax_to_nose_eyes_ratio_min=global_nose_thorax_to_nose_eyes_ratio_min
    nose_thorax_to_nose_eyes_ratio_max=global_nose_thorax_to_nose_eyes_ratio_max
    nose_thorax_to_nose_eyes_ratio_avg=global_nose_thorax_to_nose_eyes_ratio_avg
    ear_to_eye_nose_ratio_max=global_ear_to_eye_nose_ratio_max
    ear_to_eye_nose_ratio_min=global_ear_to_eye_nose_ratio_min
    
    shape=img.shape
    min_img_x=round(0.04*shape[1])
    min_img_y=round(0.04*shape[0])
    max_img_x=round(0.96*shape[1])
    max_img_y=round(0.96*shape[0])
    
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
        pose=detector.findPose(rembg_image,draw=False)
    
        
    
    with lock:
        pose=detector.findPose(rembg_image,draw=False)
    
        
    
    lmList, bboxInfo = detector.findPosition(pose,draw=False, bboxWithHands=False)
    # pp.pprint("Results from lmList:")
    # pp.pprint(lmList)
    if (use_cv_pose_detector==True): 
        print("Using CVZONE pose detector",file=sys.stderr, flush=True)
        positions.update({
        "left_eye" : lmList[3][:2],
        "right_eye": lmList[6][:2],
        "nose" : lmList[0][:2],
        "left_shoulder" : lmList[11][:2],
        "right_shoulder" : lmList[12][:2],
        "left_ear":lmList[7][:2],
        "right_ear":lmList[8][:2],
        })
    else:
        positions.update ({
        "op_left_eye" : lmList[3][:2],
        "op_right_eye": lmList[6][:2],
        "op_nose" : lmList[0][:2],
        "op_left_shoulder" : lmList[11][:2],
        "op_right_shoulder" : lmList[12][:2],
        "op_left_ear":lmList[7][:2],
        "op_right_ear":lmList[8][:2],
        "op_thorax_midpoint":[round((lmList[11][0]+lmList[12][0])/2),round((lmList[11][1]+lmList[12][1])/2)]
        })
           
   
        
    #we get normalized z depth of shoulders too for reference, we use original img here
    #image with background is better able to detect z depth
    with PoseLandmarker.create_from_options(mp_options) as landmarker:
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        image_height, image_width, _ = rgb_image.shape
        with lock:
            pose_landmarker_result = landmarker.detect(mp_image)
        # print(pose_landmarker_result,file=sys.stderr, flush=True)
        mp_pose_landmark_list={}
        mp_pose_image_landmark_list={}
        for index, elem in enumerate(POSEDETECTOR_BODY_PARTS):
                x=round(pose_landmarker_result.pose_landmarks[0][index].x*image_width)
                y=round(pose_landmarker_result.pose_landmarks[0][index].y*image_height)
                z=pose_landmarker_result.pose_landmarks[0][index].z
                # print(elem,":",pose_landmarker_result.pose_landmarks[0][index])
                mp_pose_landmark_list[elem]=[]
                mp_pose_landmark_list[elem]=[x,y,z]
                mp_pose_image_landmark_list[elem]=[x,y]
        positions.update({
            "mp_left_right_shoulder_z_distance":mp_pose_landmark_list["left shoulder"][2]-mp_pose_landmark_list["right shoulder"][2],
            "mp_left_right_ear_z_distance":mp_pose_landmark_list["left ear"][2]-mp_pose_landmark_list["right ear"][2],
            })
        # print("MP_Positions",file=sys.stderr, flush=True)
        # print(mp_pose_landmark_list,file=sys.stderr, flush=True)
        
        
        if (use_cv_pose_detector==False): 
            print("Using Mediapipe pose detector",file=sys.stderr, flush=True)
            positions.update({
            "left_eye" : mp_pose_image_landmark_list["left eye"],
            "right_eye": mp_pose_image_landmark_list["right eye"],
            "nose" : mp_pose_image_landmark_list["nose"],
            "left_shoulder" : mp_pose_image_landmark_list["left shoulder"],
            "right_shoulder" : mp_pose_image_landmark_list["right shoulder"],
            "left_ear":mp_pose_image_landmark_list["left ear"],
            "right_ear":mp_pose_image_landmark_list["right ear"],
            })
        else:
            positions.update({
            "op_left_eye" : mp_pose_image_landmark_list["left eye"],
            "op_right_eye": mp_pose_image_landmark_list["right eye"],
            "op_nose" : mp_pose_image_landmark_list["nose"],
            "op_left_shoulder" : mp_pose_image_landmark_list["left shoulder"],
            "op_right_shoulder" : mp_pose_image_landmark_list["right shoulder"],
            "op_left_ear":mp_pose_image_landmark_list["left ear"],
            "op_right_ear":mp_pose_image_landmark_list["right ear"],
            "op_thorax_midpoint":[round((mp_pose_image_landmark_list["left shoulder"][0]+mp_pose_image_landmark_list["right shoulder"][0])/2),round((mp_pose_image_landmark_list["left shoulder"][1]+mp_pose_image_landmark_list["right shoulder"][1])/2)]
            })
        print("Positions",file=sys.stderr, flush=True)
        print(positions,file=sys.stderr, flush=True)
        # if we need to ensure shoulders are aligned in z plane - we return error if they are not aligned
        if (force_shoulder_z_alignment==True and (abs(positions["mp_left_right_shoulder_z_distance"])>normalized_shoulders_z_limit)):
            print("Error!! Z values of left and right shoulder points not aligned:",file=sys.stderr, flush=True)
            print(positions["mp_left_right_shoulder_z_distance"],file=sys.stderr, flush=True)
            raise Exception('force_shoulder_z_alignment')
        if (force_shoulder_z_alignment==True and (abs(positions["mp_left_right_ear_z_distance"])>normalized_ears_z_limit)):
            print("Error!! Z values of left and right Ear points not aligned:",abs(positions["mp_left_right_ear_z_distance"],file=sys.stderr, flush=True))
            raise Exception('force_face_z_alignment')
        
    
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
    orig_shoulder_distance=shoulder_points_distance
    ear_distance=math.dist(xy_coordinate_positions["left_ear"],xy_coordinate_positions["right_ear"])
    xy_coordinate_positions["ear_distance"]=ear_distance
    xy_coordinate_positions["face_nose_thorax_distance"]=face_nose_thorax_distance
    eye_nose_distance=math.dist(xy_coordinate_positions["nose"],xy_coordinate_positions["eye_midpoint"])
    xy_coordinate_positions["eye_nose_distance"]=eye_nose_distance
    orig_eye_nose_distance=eye_nose_distance
    shoulder_to_nose_eyes_ratio=xy_coordinate_positions["shoulder_points_distance"]/xy_coordinate_positions["eye_nose_distance"]
    xy_coordinate_positions["shoulder_to_nose_eyes_ratio"]=shoulder_to_nose_eyes_ratio
    print("EyeNose Distance:"+str(eye_nose_distance),file=sys.stderr, flush=True)
    print("Ear Distance:"+str(ear_distance),file=sys.stderr, flush=True)
    print("face_nose_thorax_distance Distance:"+str(face_nose_thorax_distance),file=sys.stderr, flush=True)
    thorax_nose_to_eye_nose_ratio=xy_coordinate_positions["face_nose_thorax_distance"]/xy_coordinate_positions["eye_nose_distance"]
    xy_coordinate_positions["thorax_nose_to_eye_nose_ratio"]=thorax_nose_to_eye_nose_ratio
    print ("Thorax Nose to EyeNose Ratio:"+str(thorax_nose_to_eye_nose_ratio),file=sys.stderr, flush=True)
    ear_to_eye_nose_ratio=xy_coordinate_positions["ear_distance"]/xy_coordinate_positions["eye_nose_distance"]
    print ("Ear to EyeNose to Ratio:"+str(ear_to_eye_nose_ratio),file=sys.stderr, flush=True)
    nose_slope,nose_intercept=slope_intercept(xy_coordinate_positions["nose"],xy_coordinate_positions["thorax_midpoint"])
    xy_coordinate_positions["nose_slope"]=nose_slope
    temple_slope,temple_intercept=slope_intercept(xy_coordinate_positions["eye_midpoint"],xy_coordinate_positions["thorax_midpoint"])
    xy_coordinate_positions["temple_slope"]=temple_slope
    
    xy_coordinate_positions["orig_eye_nose_distance"]=xy_coordinate_positions["eye_nose_distance"]
    xy_coordinate_positions["orig_ear_to_eye_nose_ratio"]=ear_to_eye_nose_ratio
    xy_coordinate_positions["orig_thorax_nose_to_eye_nose_ratio"]=thorax_nose_to_eye_nose_ratio
    if (ear_to_eye_nose_ratio>ear_to_eye_nose_ratio_max
        ):
        xy_coordinate_positions["eye_nose_distance"]=xy_coordinate_positions["ear_distance"]/ear_to_eye_nose_ratio_max
        eye_nose_distance=xy_coordinate_positions["eye_nose_distance"]
        print("Reset eye_nose_distance Distance:"+str(eye_nose_distance),file=sys.stderr, flush=True)
        print ("Old Thorax to EyeNose Ratio Value:"+str(thorax_nose_to_eye_nose_ratio),file=sys.stderr, flush=True)
        thorax_nose_to_eye_nose_ratio=xy_coordinate_positions["face_nose_thorax_distance"]/xy_coordinate_positions["eye_nose_distance"]
        xy_coordinate_positions["thorax_nose_to_eye_nose_ratio"]=thorax_nose_to_eye_nose_ratio
        print ("Reset Thorax Nose to EyeNose Ratio:"+str(thorax_nose_to_eye_nose_ratio),file=sys.stderr, flush=True)
    
        
    xy_coordinate_positions["thorax_midpoint"]=reset_thorax_midpoint(thorax_nose_to_eye_nose_ratio,xy_coordinate_positions["eye_nose_distance"],xy_coordinate_positions["thorax_midpoint"],nose_slope,xy_coordinate_positions["nose"])
    thorax_midpoint=xy_coordinate_positions["thorax_midpoint"]
    face_nose_thorax_distance=math.dist(xy_coordinate_positions["nose"],xy_coordinate_positions["thorax_midpoint"])
    xy_coordinate_positions["face_nose_thorax_distance"]=face_nose_thorax_distance
    thorax_nose_to_eye_nose_ratio=xy_coordinate_positions["face_nose_thorax_distance"]/xy_coordinate_positions["eye_nose_distance"]
    xy_coordinate_positions["thorax_nose_to_eye_nose_ratio"]=thorax_nose_to_eye_nose_ratio
    
    
    
    
    if (xy_coordinate_positions["face_nose_thorax_distance"]>nose_thorax_to_nose_eyes_ratio_max*xy_coordinate_positions["eye_nose_distance"]
        ):
        print("NoseThorax to EyeNose ratio over limit, reseting nose thorax distance:",file=sys.stderr, flush=True)
        print ("Old Value:"+str(xy_coordinate_positions["face_nose_thorax_distance"]),file=sys.stderr, flush=True)
        if (xy_coordinate_positions["thorax_nose_to_eye_nose_ratio"]>=1.75*nose_thorax_to_nose_eyes_ratio_max):
            print("NoseThorax to EyeNose ratio over 1.75 limit",file=sys.stderr, flush=True)
            # face_nose_thorax_distance=0.9*1.5*nose_thorax_to_nose_eyes_ratio_max*xy_coordinate_positions["eye_nose_distance"]
            xy_coordinate_positions["thorax_midpoint"]=reset_thorax_midpoint(0.9*1.75*nose_thorax_to_nose_eyes_ratio_max,xy_coordinate_positions["eye_nose_distance"],xy_coordinate_positions["thorax_midpoint"],nose_slope,xy_coordinate_positions["nose"])
            thorax_midpoint=xy_coordinate_positions["thorax_midpoint"]
            face_nose_thorax_distance=1.75*nose_thorax_to_nose_eyes_ratio_avg*xy_coordinate_positions["eye_nose_distance"]
            
        elif (xy_coordinate_positions["thorax_nose_to_eye_nose_ratio"]>=1.55*nose_thorax_to_nose_eyes_ratio_max):
            print("NoseThorax to EyeNose ratio over 1.55 limit",file=sys.stderr, flush=True)
            # face_nose_thorax_distance=0.9*1.5*nose_thorax_to_nose_eyes_ratio_max*xy_coordinate_positions["eye_nose_distance"]
            xy_coordinate_positions["thorax_midpoint"]=reset_thorax_midpoint(0.9*1.55*nose_thorax_to_nose_eyes_ratio_max,xy_coordinate_positions["eye_nose_distance"],xy_coordinate_positions["thorax_midpoint"],nose_slope,xy_coordinate_positions["nose"])
            thorax_midpoint=xy_coordinate_positions["thorax_midpoint"]
            face_nose_thorax_distance=1.5*nose_thorax_to_nose_eyes_ratio_max*xy_coordinate_positions["eye_nose_distance"]
            
        elif (xy_coordinate_positions["thorax_nose_to_eye_nose_ratio"]>=1.5*nose_thorax_to_nose_eyes_ratio_max):
            print("NoseThorax to EyeNose ratio over 1.5 limit",file=sys.stderr, flush=True)
            # face_nose_thorax_distance=0.9*1.5*nose_thorax_to_nose_eyes_ratio_max*xy_coordinate_positions["eye_nose_distance"]
            xy_coordinate_positions["thorax_midpoint"]=reset_thorax_midpoint(0.9*1.5*nose_thorax_to_nose_eyes_ratio_max,xy_coordinate_positions["eye_nose_distance"],xy_coordinate_positions["thorax_midpoint"],nose_slope,xy_coordinate_positions["nose"])
            thorax_midpoint=xy_coordinate_positions["thorax_midpoint"]
            face_nose_thorax_distance=1.3*nose_thorax_to_nose_eyes_ratio_avg*xy_coordinate_positions["eye_nose_distance"]
            
        elif (xy_coordinate_positions["thorax_nose_to_eye_nose_ratio"]>=1.3*nose_thorax_to_nose_eyes_ratio_max):
            print("NoseThorax to EyeNose ratio less than 1.5 and over 1.3 limit",file=sys.stderr, flush=True)
            # face_nose_thorax_distance=0.95*1.25*nose_thorax_to_nose_eyes_ratio_max*xy_coordinate_positions["eye_nose_distance"]
            xy_coordinate_positions["thorax_midpoint"]=reset_thorax_midpoint(1.3*nose_thorax_to_nose_eyes_ratio_max,xy_coordinate_positions["eye_nose_distance"],xy_coordinate_positions["thorax_midpoint"],nose_slope,xy_coordinate_positions["nose"])
            thorax_midpoint=xy_coordinate_positions["thorax_midpoint"]
            face_nose_thorax_distance=1.25*nose_thorax_to_nose_eyes_ratio_avg*xy_coordinate_positions["eye_nose_distance"]
        
        elif (xy_coordinate_positions["thorax_nose_to_eye_nose_ratio"]>=1.15*nose_thorax_to_nose_eyes_ratio_max):
            print("NoseThorax to EyeNose ratio less than 1.3 and over 1.15 limit",file=sys.stderr, flush=True)
            # face_nose_thorax_distance=0.95*1.25*nose_thorax_to_nose_eyes_ratio_max*xy_coordinate_positions["eye_nose_distance"]
            xy_coordinate_positions["thorax_midpoint"]=reset_thorax_midpoint(1.15*nose_thorax_to_nose_eyes_ratio_avg,xy_coordinate_positions["eye_nose_distance"],xy_coordinate_positions["thorax_midpoint"],nose_slope,xy_coordinate_positions["nose"])
            thorax_midpoint=xy_coordinate_positions["thorax_midpoint"]
            face_nose_thorax_distance=1.23*nose_thorax_to_nose_eyes_ratio_avg*xy_coordinate_positions["eye_nose_distance"]
        
        elif (xy_coordinate_positions["thorax_nose_to_eye_nose_ratio"]>=1.09*nose_thorax_to_nose_eyes_ratio_max):
            print("NoseThorax to EyeNose ratio less than 1.15 and over 1.09 limit",file=sys.stderr, flush=True)
            # face_nose_thorax_distance=0.95*1.25*nose_thorax_to_nose_eyes_ratio_max*xy_coordinate_positions["eye_nose_distance"]
            xy_coordinate_positions["thorax_midpoint"]=reset_thorax_midpoint(1.09*nose_thorax_to_nose_eyes_ratio_max,xy_coordinate_positions["eye_nose_distance"],xy_coordinate_positions["thorax_midpoint"],nose_slope,xy_coordinate_positions["nose"])
            thorax_midpoint=xy_coordinate_positions["thorax_midpoint"]
            face_nose_thorax_distance=1.09*nose_thorax_to_nose_eyes_ratio_avg*xy_coordinate_positions["eye_nose_distance"]
        
        else:
            print("NoseThorax to EyeNose ratio less than 1.05 limit",file=sys.stderr, flush=True)
            # xy_coordinate_positions["thorax_midpoint"]=reset_thorax_midpoint(nose_thorax_to_nose_eyes_ratio_max,xy_coordinate_positions["eye_nose_distance"],xy_coordinate_positions["thorax_midpoint"],nose_slope,xy_coordinate_positions["nose"])
            # thorax_midpoint=xy_coordinate_positions["thorax_midpoint"]
            face_nose_thorax_distance=nose_thorax_to_nose_eyes_ratio_avg*xy_coordinate_positions["eye_nose_distance"] 
        
    elif (xy_coordinate_positions["face_nose_thorax_distance"]<nose_thorax_to_nose_eyes_ratio_min*xy_coordinate_positions["eye_nose_distance"]
        ):
        print("NoseThorax to EyeNose ratio under limit, reseting nose thorax distance:",file=sys.stderr, flush=True)
        print ("Old Face Nose thorax Distance Value:"+str(xy_coordinate_positions["face_nose_thorax_distance"]),file=sys.stderr, flush=True)
        # thorax_nose_to_eye_nose_ratio=xy_coordinate_positions["face_nose_thorax_distance"]/xy_coordinate_positions["eye_nose_distance"]
        print ("Old Thorax nose to Eye nose ratio:"+str(thorax_nose_to_eye_nose_ratio),file=sys.stderr,flush=True)
        print ("Original Nose Thorax to Eye Nose Ratio"+str(xy_coordinate_positions["orig_thorax_nose_to_eye_nose_ratio"]),file=sys.stderr,flush=True)
        multiplier=xy_coordinate_positions["orig_thorax_nose_to_eye_nose_ratio"]/thorax_nose_to_eye_nose_ratio
        print ("Multiplier:"+str(multiplier),file=sys.stderr,flush=True)
        orig_multiplier=multiplier
        if (multiplier<0.8) :
            multiplier=1.15
            xy_coordinate_positions["thorax_midpoint"]=reset_thorax_midpoint(nose_thorax_to_nose_eyes_ratio_min*multiplier,xy_coordinate_positions["eye_nose_distance"],xy_coordinate_positions["thorax_midpoint"],nose_slope,xy_coordinate_positions["nose"])
            multiplier=orig_multiplier
        elif (multiplier<=1.15) :
            multiplier=1.12
            xy_coordinate_positions["thorax_midpoint"]=reset_thorax_midpoint(nose_thorax_to_nose_eyes_ratio_min*multiplier,xy_coordinate_positions["eye_nose_distance"],xy_coordinate_positions["thorax_midpoint"],nose_slope,xy_coordinate_positions["nose"])
            multiplier=orig_multiplier
        else:
            multiplier=1
            xy_coordinate_positions["thorax_midpoint"]=reset_thorax_midpoint(nose_thorax_to_nose_eyes_ratio_avg*multiplier,xy_coordinate_positions["eye_nose_distance"],xy_coordinate_positions["thorax_midpoint"],nose_slope,xy_coordinate_positions["nose"])
            multiplier=orig_multiplier
            
        thorax_midpoint=xy_coordinate_positions["thorax_midpoint"]
        # face_nose_thorax_distance=math.dist(xy_coordinate_positions["nose"],xy_coordinate_positions["thorax_midpoint"])
        face_nose_thorax_distance=nose_thorax_to_nose_eyes_ratio_min*multiplier*xy_coordinate_positions["eye_nose_distance"] 
        xy_coordinate_positions["face_nose_thorax_distance"]=face_nose_thorax_distance
        print ("New face nose thorax distance Value:"+str(xy_coordinate_positions["face_nose_thorax_distance"]),file=sys.stderr, flush=True)
        thorax_nose_to_eye_nose_ratio=xy_coordinate_positions["face_nose_thorax_distance"]/xy_coordinate_positions["eye_nose_distance"]
        xy_coordinate_positions["thorax_nose_to_eye_nose_ratio"]=thorax_nose_to_eye_nose_ratio
        print ("New Thorax Nose to EyeNose Ratio:"+str(thorax_nose_to_eye_nose_ratio),file=sys.stderr, flush=True)
        face_nose_thorax_distance=nose_thorax_to_nose_eyes_ratio_avg*xy_coordinate_positions["eye_nose_distance"]
   
    elif (xy_coordinate_positions["face_nose_thorax_distance"]<nose_thorax_to_nose_eyes_ratio_avg*xy_coordinate_positions["eye_nose_distance"]
        ):
        print("NoseThorax to EyeNose ratio under avg, reseting nose thorax distance:",file=sys.stderr, flush=True)
        if (thorax_nose_to_eye_nose_ratio>((nose_thorax_to_nose_eyes_ratio_avg+nose_thorax_to_nose_eyes_ratio_min)/2)):
            print("NoseThorax to EyeNose requires thorax midpoint shift",file=sys.stderr, flush=True)
            xy_coordinate_positions["thorax_midpoint"]=reset_thorax_midpoint(nose_thorax_to_nose_eyes_ratio_avg,xy_coordinate_positions["eye_nose_distance"],xy_coordinate_positions["thorax_midpoint"],nose_slope,xy_coordinate_positions["nose"])
            thorax_midpoint=xy_coordinate_positions["thorax_midpoint"]
        
        
        xy_coordinate_positions["face_nose_thorax_distance"]=face_nose_thorax_distance
        thorax_nose_to_eye_nose_ratio=xy_coordinate_positions["face_nose_thorax_distance"]/xy_coordinate_positions["eye_nose_distance"]
        xy_coordinate_positions["thorax_nose_to_eye_nose_ratio"]=thorax_nose_to_eye_nose_ratio
        print ("New Thorax Nose to EyeNose Ratio:"+str(thorax_nose_to_eye_nose_ratio),file=sys.stderr, flush=True)


    xy_coordinate_positions["face_nose_thorax_distance"]=face_nose_thorax_distance
    print ("New face nose thorax distance Value:"+str(xy_coordinate_positions["face_nose_thorax_distance"]),file=sys.stderr, flush=True)
    shoulder_to_nose_eyes_ratio=xy_coordinate_positions["shoulder_points_distance"]/xy_coordinate_positions["eye_nose_distance"]
    xy_coordinate_positions["shoulder_to_nose_eyes_ratio"]=shoulder_to_nose_eyes_ratio
    print("Shoulder to EyeNose ratio:"+str(shoulder_to_nose_eyes_ratio),file=sys.stderr, flush=True)
    if (xy_coordinate_positions["shoulder_to_nose_eyes_ratio"]>shoulder_to_nose_eyes_ratio_max):
        print("Shoulder to EyeNose ratio over limit, reseting sholder point distance",file=sys.stderr, flush=True)
        xy_coordinate_positions["shoulder_points_distance"]=xy_coordinate_positions["eye_nose_distance"]*shoulder_to_nose_eyes_ratio_max
        shoulder_to_nose_eyes_ratio=xy_coordinate_positions["shoulder_points_distance"]/xy_coordinate_positions["eye_nose_distance"]
        xy_coordinate_positions["shoulder_to_nose_eyes_ratio"]=shoulder_to_nose_eyes_ratio
        print("New Shoulder to EyeNose ratio:"+str(shoulder_to_nose_eyes_ratio),file=sys.stderr, flush=True)
        
    print("XY Positions",file=sys.stderr, flush=True)
    print(xy_coordinate_positions,file=sys.stderr, flush=True)
    
    # if (round((xy_coordinate_positions["shoulder_points_distance"]/2)<round(0.95*xy_coordinate_positions["face_nose_thorax_distance"]))
        # and ((abs(math.degrees(math.atan(nose_slope)))>=degrees_nose_slope_min and abs(math.degrees(math.atan(nose_slope))) <=degrees_nose_slope_max ))):
        # print ("Too narrow on shoulder points,setting to nose_thorax",file=sys.stderr, flush=True)
        # xy_coordinate_positions["shoulder_points_distance"]=round(2*face_nose_thorax_distance)
    
    if use_different_horizontal_vertical_scale==True:
        
        xy_coordinate_positions["horizontal_reduced_circle_radius"]=round(xy_coordinate_positions["shoulder_points_distance"]/2* horizontal_ratio/100)
        xy_coordinate_positions["vertical_reduced_circle_radius"]=round(xy_coordinate_positions["shoulder_points_distance"]/2*horizontal_ratio/100)
    else:
        xy_coordinate_positions["vertical_reduced_circle_radius"]=round(xy_coordinate_positions["face_nose_thorax_distance"] * vertical_ratio/100)
        xy_coordinate_positions["horizontal_reduced_circle_radius"]=xy_coordinate_positions["vertical_reduced_circle_radius"]
    print("Vertical Radius:"+str(xy_coordinate_positions["vertical_reduced_circle_radius"]),file=sys.stderr, flush=True)
    print("Horizontal Radius:"+str(xy_coordinate_positions["horizontal_reduced_circle_radius"]),file=sys.stderr, flush=True)
    xy_coordinate_positions["vertical_ratio"]=xy_coordinate_positions["vertical_reduced_circle_radius"]/xy_coordinate_positions["horizontal_reduced_circle_radius"]
    xy_coordinate_positions["horizontal_ratio"]=xy_coordinate_positions["horizontal_reduced_circle_radius"]/xy_coordinate_positions["vertical_reduced_circle_radius"]
    
    print("Vertical Ratio:"+str(xy_coordinate_positions["vertical_ratio"]),file=sys.stderr, flush=True)
    print("Horizontal Ratio:"+str(xy_coordinate_positions["horizontal_ratio"]),file=sys.stderr, flush=True)
    
    # if we are too wide on the vertical scale or horizontal scale- we tie it to horizontal scale
    if ((xy_coordinate_positions["vertical_reduced_circle_radius"]/xy_coordinate_positions["horizontal_reduced_circle_radius"])>max_vertical_horizontal_ratio
        or (xy_coordinate_positions["horizontal_reduced_circle_radius"]/xy_coordinate_positions["vertical_reduced_circle_radius"])>max_horizontal_vertical_ratio
        ):
        print("Too wide on ratio",file=sys.stderr, flush=True)
        xy_coordinate_positions["vertical_reduced_circle_radius"]=round(xy_coordinate_positions["face_nose_thorax_distance"] * vertical_ratio/100)
        xy_coordinate_positions["horizontal_reduced_circle_radius"]=xy_coordinate_positions["vertical_reduced_circle_radius"] * max_horizontal_vertical_ratio
        print("Max Horizontal Vertical Ratio:"+str(max_horizontal_vertical_ratio),file=sys.stderr, flush=True)
        xy_coordinate_positions["vertical_ratio"]=xy_coordinate_positions["vertical_reduced_circle_radius"]/xy_coordinate_positions["horizontal_reduced_circle_radius"]
        xy_coordinate_positions["horizontal_ratio"]=xy_coordinate_positions["horizontal_reduced_circle_radius"]/xy_coordinate_positions["vertical_reduced_circle_radius"]
        print("New Vertical Ratio:"+str(xy_coordinate_positions["vertical_ratio"]),file=sys.stderr, flush=True)
        print("New Horizontal Ratio:"+str(xy_coordinate_positions["horizontal_ratio"]),file=sys.stderr, flush=True)
    
    
    
    vertical_reduced_circle_radius=xy_coordinate_positions["vertical_reduced_circle_radius"]
    horizontal_reduced_circle_radius=xy_coordinate_positions["horizontal_reduced_circle_radius"]
    
    xy_coordinate_positions["reduced_circle_radius"]=round(xy_coordinate_positions["face_nose_thorax_distance"] * vertical_ratio/100)
    reduced_circle_radius=xy_coordinate_positions["reduced_circle_radius"]
    xy_coordinate_positions["thorax_top_bottom_distance"]=xy_coordinate_positions["horizontal_reduced_circle_radius"]*2
    nose_slope,nose_intercept=slope_intercept(xy_coordinate_positions["nose"],xy_coordinate_positions["thorax_midpoint"])
    shoulder_slope,shoulder_intercept=slope_intercept(xy_coordinate_positions["left_shoulder"],xy_coordinate_positions["right_shoulder"])
    eye_slope,eye_intercept=slope_intercept(xy_coordinate_positions["left_eye"],xy_coordinate_positions["right_eye"])
    ear_slope,ear_intercept=slope_intercept(xy_coordinate_positions["left_ear"],xy_coordinate_positions["right_ear"])
    
        
    xy_coordinate_positions["temple_slope"]=temple_slope
    xy_coordinate_positions["nose_slope"]=nose_slope
    xy_coordinate_positions["shoulder_slope"]=shoulder_slope
    xy_coordinate_positions["eye_slope"]=eye_slope
    xy_coordinate_positions["ear_slope"]=ear_slope
    orig_ear_slope=ear_slope
    
    
    
    
    
    #we align with nose thorax slope if it is off upto 10 degrees rather than shoulder slope 
    # if shoulder slope >4
    print ("Original Nose Slope",file=sys.stderr, flush=True)
    print(math.degrees(math.atan(xy_coordinate_positions["nose_slope"])),file=sys.stderr, flush=True)
    print ("Original Temple Slope",file=sys.stderr, flush=True)
    print(math.degrees(math.atan(xy_coordinate_positions["temple_slope"])),file=sys.stderr, flush=True)
    print ("Original Shoulder Slope",file=sys.stderr, flush=True)
    print(math.degrees(math.atan(xy_coordinate_positions["shoulder_slope"])),file=sys.stderr, flush=True)
    print ("Original Eye Slope",file=sys.stderr, flush=True)
    print(math.degrees(math.atan(xy_coordinate_positions["eye_slope"])),file=sys.stderr, flush=True)
    print ("Original Ear Slope",file=sys.stderr, flush=True)
    print(math.degrees(math.atan(xy_coordinate_positions["ear_slope"])),file=sys.stderr, flush=True)
    
        
    orig_nose_slope=xy_coordinate_positions["nose_slope"]
    orig_shoulder_slope=xy_coordinate_positions["shoulder_slope"]
    if (
        ((90-abs(math.degrees(math.atan(xy_coordinate_positions["temple_slope"]))))<=(90-abs(math.degrees(math.atan(xy_coordinate_positions["nose_slope"])))))
        and (abs(math.degrees(math.atan(nose_slope)))<degrees_nose_slope_min)
        ):
        print("Temple closer to vertical than Nose, using Temple Slope as Nose Slope",file=sys.stderr, flush=True)
        nose_slope=temple_slope
        xy_coordinate_positions["nose_slope"]=temple_slope
    
    if ((abs(math.degrees(math.atan(shoulder_slope)))<=degrees_shoulder_slope_max ) 
        #and (abs(math.degrees(math.atan(shoulder_slope)))<=degrees_shoulder_slope_max and abs(math.degrees(math.atan(ear_slope)))<=degrees_shoulder_slope_max)
        and 
        (
        (xy_coordinate_positions["shoulder_slope"]*xy_coordinate_positions["ear_slope"]>0)
        # or (xy_coordinate_positions["shoulder_slope"]*xy_coordinate_positions["ear_slope"]<0 
           # and (
           # (abs(math.degrees(math.atan(xy_coordinate_positions["ear_slope"])))<=degrees_shoulder_slope_max*0.8)
           # or (abs(math.degrees(math.atan(xy_coordinate_positions["shoulder_slope"])))<degrees_shoulder_slope_max*0.5)
           # )
           # )  # if shoulder_slope is shallow - follow shoulder slope
        )) :
       print ("Following shoulder slope and trying to Reset nose slope as shoulder and ear slope is horizontal and same inclines or shoulder slope is shallow",file=sys.stderr, flush=True)
       if (math.degrees((math.atan(nose_slope))>=0) and math.degrees((math.atan(nose_slope))<=90)):
           nose_slope=math.tan(math.radians(math.degrees(math.atan(shoulder_slope)) - 90))
       else:
           nose_slope=math.tan(math.radians(90+ math.degrees(math.atan(shoulder_slope))))
       
       if ((90-abs(math.degrees(math.atan(nose_slope))))<(90-abs(math.degrees(math.atan(orig_nose_slope))))):
            xy_coordinate_positions["nose_slope"]=nose_slope
            print("Reset Nose slope",file=sys.stderr, flush=True)
            print(math.degrees(math.atan(xy_coordinate_positions['nose_slope'])),file=sys.stderr, flush=True)
       else:
           xy_coordinate_positions["nose_slope"]=orig_nose_slope
           
    elif ((abs(math.degrees(math.atan(shoulder_slope)))<=degrees_shoulder_slope_max ) 
        and (xy_coordinate_positions["shoulder_slope"]*xy_coordinate_positions["ear_slope"]<0)
        ) :
       if (abs(math.degrees(math.atan(shoulder_slope)))<=degrees_shoulder_slope_max and abs(math.degrees(math.atan(ear_slope)))<=degrees_shoulder_slope_max
           and (
           (xy_coordinate_positions["shoulder_slope"]*xy_coordinate_positions["ear_slope"]>=0)
           or (xy_coordinate_positions["shoulder_slope"]*xy_coordinate_positions["ear_slope"]<0 
           and (
           (abs(math.degrees(math.atan(xy_coordinate_positions["ear_slope"])))<=degrees_shoulder_slope_max*0.8)
           )
           )
           )):
           print ("Following Shoulder slope and trying to Reset nose slope as shoulder slope is in limit but shoulder and ear have different inclines and shoulder and ear slope in limits",file=sys.stderr, flush=True) 
           if (math.degrees((math.atan(nose_slope))>=0) and math.degrees((math.atan(nose_slope))<=90)):
               nose_slope=math.tan(math.radians(math.degrees(math.atan(shoulder_slope)) - 90))
           else:
               nose_slope=math.tan(math.radians(90+ math.degrees(math.atan(shoulder_slope))))
           
           if ((90-abs(math.degrees(math.atan(nose_slope))))<(90-abs(math.degrees(math.atan(orig_nose_slope))))):
                xy_coordinate_positions["nose_slope"]=nose_slope
                print("Reset Nose slope",file=sys.stderr, flush=True)
                print(math.degrees(math.atan(xy_coordinate_positions['nose_slope'])),file=sys.stderr, flush=True)
           else:
               xy_coordinate_positions["nose_slope"]=orig_nose_slope
               
       elif ( (math.degrees((math.atan(nose_slope))<0) 
                   and (math.degrees(math.atan(shoulder_slope))>=0)
                   )
                   or (math.degrees((math.atan(nose_slope))>=0) and math.degrees((math.atan(nose_slope))<=90)
                   and (math.degrees(math.atan(shoulder_slope))<0 ))
              ):
               print ("Following Shoulder slope and trying to Reset nose slope as shoulder slope is in limit and shoulder and nose have different inclines",file=sys.stderr, flush=True) 
               
               if (math.degrees((math.atan(nose_slope))>=0) and math.degrees((math.atan(nose_slope))<=90)):
                   nose_slope=math.tan(math.radians(math.degrees(math.atan(shoulder_slope)) - 90))
               else:
                   nose_slope=math.tan(math.radians(90+ math.degrees(math.atan(shoulder_slope))))
               
               if ((90-abs(math.degrees(math.atan(nose_slope))))<(90-abs(math.degrees(math.atan(orig_nose_slope))))):
                    xy_coordinate_positions["nose_slope"]=nose_slope
                    print("Reset Nose slope",file=sys.stderr, flush=True)
                    print(math.degrees(math.atan(xy_coordinate_positions['nose_slope'])),file=sys.stderr, flush=True)
               else:
                   xy_coordinate_positions["nose_slope"]=orig_nose_slope
               
       elif (((abs(math.degrees(math.atan(nose_slope)))>=degrees_nose_slope_min +2
           and abs(math.degrees(math.atan(nose_slope))) <=degrees_nose_slope_max ))
           and (xy_coordinate_positions["shoulder_slope"]*xy_coordinate_positions["ear_slope"]<0)
           ):
           
           if (
               abs(math.degrees(math.atan(shoulder_slope))<degrees_shoulder_slope_max*0.5)
               and abs(math.degrees(math.atan(xy_coordinate_positions["ear_slope"]))>degrees_shoulder_slope_max*.9)
               ): #if shoulder and ears are almost horizontal follow them
               print ("Following Shoulder slope and trying to Reset nose slope as shoulder slope is in limit and shoulder and nose have different inclines, but shoulder and ears shallow",file=sys.stderr, flush=True) 
               
               if (math.degrees((math.atan(nose_slope))>=0) and math.degrees((math.atan(nose_slope))<=90)):
                   nose_slope=math.tan(math.radians(math.degrees(math.atan(shoulder_slope)) - 90))
               else:
                   nose_slope=math.tan(math.radians(90+ math.degrees(math.atan(shoulder_slope))))
               
               if ((90-abs(math.degrees(math.atan(nose_slope))))<(90-abs(math.degrees(math.atan(orig_nose_slope))))):
                    xy_coordinate_positions["nose_slope"]=nose_slope
                    print("Reset Nose slope",file=sys.stderr, flush=True)
                    print(math.degrees(math.atan(xy_coordinate_positions['nose_slope'])),file=sys.stderr, flush=True)
               else:
                   xy_coordinate_positions["nose_slope"]=orig_nose_slope
               
           else:
               print ("Following Nose slope and trying to Reset shoulder slope as shoulder slope is in limit but shoulder and ear have different inclines and shoulder slope or ear slope are out of limits and nose slope in limits",file=sys.stderr, flush=True)
               if (math.degrees((math.atan(nose_slope))>=0) and math.degrees((math.atan(nose_slope))<=90)):
                    shoulder_slope=math.tan(math.radians(math.degrees(abs(math.atan(nose_slope)))-90))
               else:
                   shoulder_slope=math.tan(math.radians(90-math.degrees(abs(math.atan(nose_slope)))))
                   
               if (abs(shoulder_slope)<abs(orig_shoulder_slope) or 1==1):    
                   xy_coordinate_positions["shoulder_slope"]=shoulder_slope
                   print("Reset Shoulder slope",file=sys.stderr, flush=True)
                   print(math.degrees(math.atan(xy_coordinate_positions["shoulder_slope"])),file=sys.stderr, flush=True)
               else:
                   xy_coordinate_positions["shoulder_slope"]=orig_shoulder_slope
               
               
       else:
           
               if ( (math.degrees((math.atan(nose_slope))<0) 
                   and (math.degrees(math.atan(shoulder_slope))>=0 or math.degrees(math.atan(ear_slope))>=0)
                   )
                   or (math.degrees((math.atan(nose_slope))>=0) and math.degrees((math.atan(nose_slope))<=90)
                   and (math.degrees(math.atan(shoulder_slope))<0 or math.degrees(math.atan(ear_slope))<0))
                   ):
                   
                   if (abs(math.degrees(math.atan(shoulder_slope)))<=degrees_shoulder_slope_max
                        and (
                        (xy_coordinate_positions["ear_slope"]*xy_coordinate_positions["shoulder_slope"]>=0) 
                        #or(xy_coordinate_positions["shoulder_slope"]*xy_coordinate_positions["ear_slope"]<0 and abs(math.degrees(math.atan(xy_coordinate_positions["ear_slope"])))<=degrees_shoulder_slope_max*0.8)
                        or 1==1
                        )
                       ) :  #always follow shoulder if its in limits
                       print ("Following Shoulder slope and trying to Reset nose slope as shoulder slope is in limit and shoulder and ear have same inclines and nose slope in limits and different inclined as shoulder and shoulder slope in limits",file=sys.stderr, flush=True)
                       if (math.degrees((math.atan(nose_slope))>=0) and math.degrees((math.atan(nose_slope))<=90)):
                           nose_slope=math.tan(math.radians(math.degrees(math.atan(shoulder_slope)) - 90))
                       else:
                           nose_slope=math.tan(math.radians(90+ math.degrees(math.atan(shoulder_slope))))
                       
                       if ((90-abs(math.degrees(math.atan(nose_slope))))<(90-abs(math.degrees(math.atan(orig_nose_slope))))):
                            xy_coordinate_positions["nose_slope"]=nose_slope
                            print("Reset Nose slope",file=sys.stderr, flush=True)
                            print(math.degrees(math.atan(xy_coordinate_positions['nose_slope'])),file=sys.stderr, flush=True)
                       else:
                           xy_coordinate_positions["nose_slope"]=orig_nose_slope 
                       
                   else:
                        print ("Following Nose slope and trying to Reset shoulder slope as shoulder slope is in limit  and nose slope in limits and different inclined as shoulder and shoulder slope out of limits or shoulder and ear have different slopes",file=sys.stderr, flush=True)
                        if (math.degrees((math.atan(nose_slope))>=0) and math.degrees((math.atan(nose_slope))<=90)):
                            shoulder_slope=math.tan(math.radians(math.degrees(abs(math.atan(nose_slope)))-90))
                        else:
                            shoulder_slope=math.tan(math.radians(90-math.degrees(abs(math.atan(nose_slope)))))
                       
                        if (abs(shoulder_slope)<abs(orig_shoulder_slope) or 1==1):    
                            xy_coordinate_positions["shoulder_slope"]=shoulder_slope
                            print("Reset Shoulder slope",file=sys.stderr, flush=True)
                            print(math.degrees(math.atan(xy_coordinate_positions["shoulder_slope"])),file=sys.stderr, flush=True)
                        else:
                            xy_coordinate_positions["shoulder_slope"]=orig_shoulder_slope
                   
                   
                   
               else:
                   if (abs(math.degrees((math.atan(nose_slope))))>(90-2*vertical_offset) 
                   and abs(math.degrees((math.atan(nose_slope))))<=90):
                        print ("Following Nose slope and trying to Reset shoulder slope as shoulder slope is in limit but shoulder and ear have different inclines but nose and shoulder have same incline and  nose slope out of limits and nose above 70deg",file=sys.stderr, flush=True)       
                       
                        if (math.degrees((math.atan(nose_slope))>=0) and math.degrees((math.atan(nose_slope))<=90)):
                            shoulder_slope=math.tan(math.radians(math.degrees(abs(math.atan(nose_slope)))-90))
                        else:
                            shoulder_slope=math.tan(math.radians(90-math.degrees(abs(math.atan(nose_slope)))))
                       
                        if (abs(shoulder_slope)<abs(orig_shoulder_slope) or 1==1):    
                            xy_coordinate_positions["shoulder_slope"]=shoulder_slope
                            print("Reset Shoulder slope",file=sys.stderr, flush=True)
                            print(math.degrees(math.atan(xy_coordinate_positions["shoulder_slope"])),file=sys.stderr, flush=True)
                        else:
                            xy_coordinate_positions["shoulder_slope"]=orig_shoulder_slope
                   else:    
                           print ("Following Ear slope and trying to Reset shoulder slope as shoulder slope is in limit but shoulder and ear have different inclines but nose and shoulder have same incline and  nose slope out of limits and nose below 70deg",file=sys.stderr, flush=True)
                           shoulder_slope=ear_slope
                           xy_coordinate_positions["shoulder_slope"]=ear_slope
                           if (math.degrees((math.atan(nose_slope))>=0) and math.degrees((math.atan(nose_slope))<=90)):
                               nose_slope=math.tan(math.radians(math.degrees(math.atan(shoulder_slope)) - 90))
                           else:
                               nose_slope=math.tan(math.radians(90+ math.degrees(math.atan(shoulder_slope))))
                           
                           if ((90-abs(math.degrees(math.atan(nose_slope))))<(90-abs(math.degrees(math.atan(orig_nose_slope))))):
                                xy_coordinate_positions["nose_slope"]=nose_slope
                                print("Reset Nose slope",file=sys.stderr, flush=True)
                                print(math.degrees(math.atan(xy_coordinate_positions['nose_slope'])),file=sys.stderr, flush=True)
                           else:
                               xy_coordinate_positions["nose_slope"]=orig_nose_slope
                   
                   # print ("Following Ear slope and trying to Reset shoulder slope as shoulder slope is in limit but shoulder and ear have different inclines but nose and shoulder have same incline and  nose slope out of limits",file=sys.stderr, flush=True)
                   # shoulder_slope=ear_slope
                   # xy_coordinate_positions["shoulder_slope"]=ear_slope
                   # if (math.degrees((math.atan(nose_slope))>=0) and math.degrees((math.atan(nose_slope))<=90)):
                       # nose_slope=math.tan(math.radians(math.degrees(math.atan(shoulder_slope)) - 90))
                   # else:
                       # nose_slope=math.tan(math.radians(90+ math.degrees(math.atan(shoulder_slope))))
                   
                   # if ((90-abs(math.degrees(math.atan(nose_slope))))<(90-abs(math.degrees(math.atan(orig_nose_slope))))):
                        # xy_coordinate_positions["nose_slope"]=nose_slope
                        # print("Reset Nose slope",file=sys.stderr, flush=True)
                        # print(math.degrees(math.atan(xy_coordinate_positions['nose_slope'])),file=sys.stderr, flush=True)
                   # else:
                       # xy_coordinate_positions["nose_slope"]=orig_nose_slope
    
    elif (((abs(math.degrees(math.atan(nose_slope)))>=degrees_nose_slope_min  and abs(math.degrees(math.atan(nose_slope))) <=degrees_nose_slope_max ) 
         and (xy_coordinate_positions["ear_slope"]*xy_coordinate_positions["shoulder_slope"]>0)
         )
        ):
           if ( (math.degrees((math.atan(nose_slope))<0) 
                       and math.degrees(math.atan(shoulder_slope))>=0)
                       or (math.degrees((math.atan(nose_slope))>=0) and math.degrees((math.atan(nose_slope))<=90)
                       and math.degrees(math.atan(shoulder_slope))<0)
                       ):
                       
                       if ((90-abs(math.degrees(math.atan(xy_coordinate_positions["temple_slope"]))))<=(90-abs(math.degrees(math.atan(xy_coordinate_positions["nose_slope"]))))):
                            print ("Following Temple slope and trying to Reset shoudler slope as nose slope is vertical and shoulder and ear have same incline and shoulder and nose are in opposite incline and temple close is closer to vertical",file=sys.stderr, flush=True)
                            nose_slope=temple_slope
                            xy_coordinate_positions["nose_slope"]=temple_slope
                            if (math.degrees((math.atan(nose_slope))>=0) and math.degrees((math.atan(nose_slope))<=90)):
                                   shoulder_slope=math.tan(math.radians(math.degrees(abs(math.atan(nose_slope)))-90))
                            else:
                                   shoulder_slope=math.tan(math.radians(90-math.degrees(abs(math.atan(nose_slope)))))
                                   
                            if (abs(shoulder_slope)<abs(orig_shoulder_slope) and abs(math.degrees((math.atan(nose_slope))))>2+degrees_nose_slope_max):    
                               xy_coordinate_positions["shoulder_slope"]=shoulder_slope
                               print("Reset Shoulder slope",file=sys.stderr, flush=True)
                               print(math.degrees(math.atan(xy_coordinate_positions["shoulder_slope"])),file=sys.stderr, flush=True)
                            else:
                               xy_coordinate_positions["shoulder_slope"]=orig_shoulder_slope
               
                       else:
                            print ("Following Nose slope and trying to Reset shoudler slope as nose slope is vertical and shoulder and ear have same incline and shoulder and nose are in opposite incline",file=sys.stderr, flush=True)       
                            if (math.degrees((math.atan(nose_slope))>=0) and math.degrees((math.atan(nose_slope))<=90)):
                                shoulder_slope=math.tan(math.radians(math.degrees(abs(math.atan(nose_slope)))-90))
                            else:
                                shoulder_slope=math.tan(math.radians(90-math.degrees(abs(math.atan(nose_slope)))))
                           
                            if (abs(shoulder_slope)<abs(orig_shoulder_slope) or 1==1):    
                                xy_coordinate_positions["shoulder_slope"]=shoulder_slope
                                print("Reset Shoulder slope",file=sys.stderr, flush=True)
                                print(math.degrees(math.atan(xy_coordinate_positions["shoulder_slope"])),file=sys.stderr, flush=True)
                            else:
                                xy_coordinate_positions["shoulder_slope"]=orig_shoulder_slope
                            
           else:
               
               if (
                   (90-abs(math.degrees(math.atan(xy_coordinate_positions["temple_slope"]))))<=(90-abs(math.degrees(math.atan(xy_coordinate_positions["nose_slope"]))))
                   and (abs(math.degrees(math.atan(ear_slope)))>degrees_shoulder_slope_max-1.2)
                   ):
                            print ("Following Temple slope and trying to Reset shoulder slope as nose slope is vertical and shoulder and ear have same inclines and temple is closer to vertical",file=sys.stderr, flush=True)
                            nose_slope=temple_slope
                            xy_coordinate_positions["nose_slope"]=temple_slope
                            if (math.degrees((math.atan(nose_slope))>=0) and math.degrees((math.atan(nose_slope))<=90)):
                                   shoulder_slope=math.tan(math.radians(math.degrees(abs(math.atan(nose_slope)))-90))
                            else:
                                   shoulder_slope=math.tan(math.radians(90-math.degrees(abs(math.atan(nose_slope)))))
                                   
                            if (abs(shoulder_slope)<abs(orig_shoulder_slope) or 1==1):    
                               xy_coordinate_positions["shoulder_slope"]=shoulder_slope
                               print("Reset Shoulder slope",file=sys.stderr, flush=True)
                               print(math.degrees(math.atan(xy_coordinate_positions["shoulder_slope"])),file=sys.stderr, flush=True)
                            else:
                               xy_coordinate_positions["shoulder_slope"]=orig_shoulder_slope

               elif (abs(math.degrees(math.atan(ear_slope)))<degrees_shoulder_slope_max):
                       print ("Following Ear slope and trying to Reset shoulder slope as nose slope is vertical and shoulder and ear have same inclines and ear slope in limits",file=sys.stderr, flush=True)
                       shoulder_slope=ear_slope
                       xy_coordinate_positions["shoulder_slope"]=ear_slope
                       if (math.degrees((math.atan(nose_slope))>=0) and math.degrees((math.atan(nose_slope))<=90)):
                           nose_slope=math.tan(math.radians(math.degrees(math.atan(shoulder_slope)) - 90))
                       else:
                           nose_slope=math.tan(math.radians(90+ math.degrees(math.atan(shoulder_slope))))
                       
                       if ((90-abs(math.degrees(math.atan(nose_slope))))<(90-abs(math.degrees(math.atan(orig_nose_slope))))):
                            xy_coordinate_positions["nose_slope"]=nose_slope
                            print("Reset Nose slope",file=sys.stderr, flush=True)
                            print(math.degrees(math.atan(xy_coordinate_positions['nose_slope'])),file=sys.stderr, flush=True)
                       else:
                           xy_coordinate_positions["nose_slope"]=orig_nose_slope
               else:
                   if ((90-abs(math.degrees(math.atan(xy_coordinate_positions["temple_slope"]))))<=(90-abs(math.degrees(math.atan(xy_coordinate_positions["nose_slope"]))))):
                            print ("Following Temple slope and trying to Reset shoulder slope as nose slope is vertical and shoulder and ear have same inclines and temple is closer to vertical",file=sys.stderr, flush=True)
                            nose_slope=temple_slope
                            xy_coordinate_positions["nose_slope"]=temple_slope
                            if (math.degrees((math.atan(nose_slope))>=0) and math.degrees((math.atan(nose_slope))<=90)):
                                   shoulder_slope=math.tan(math.radians(math.degrees(abs(math.atan(nose_slope)))-90))
                            else:
                                   shoulder_slope=math.tan(math.radians(90-math.degrees(abs(math.atan(nose_slope)))))
                                   
                            if (abs(shoulder_slope)<abs(orig_shoulder_slope) or 1==1):    
                               xy_coordinate_positions["shoulder_slope"]=shoulder_slope
                               print("Reset Shoulder slope",file=sys.stderr, flush=True)
                               print(math.degrees(math.atan(xy_coordinate_positions["shoulder_slope"])),file=sys.stderr, flush=True)
                            else:
                               xy_coordinate_positions["shoulder_slope"]=orig_shoulder_slope
                   else:          
                           print ("Following Nose slope and trying to Reset shoulder slope as nose slope is vertical and shoulder and ear have same inclines",file=sys.stderr, flush=True)
                           if ((math.atan(nose_slope)>=0) and (math.atan(nose_slope)<=90)):
                                shoulder_slope=math.tan(math.radians(math.degrees(abs(math.atan(nose_slope)))-90))
                           else:
                               shoulder_slope=math.tan(math.radians(90-math.degrees(abs(math.atan(nose_slope)))))
                               
                           if (abs(shoulder_slope)<abs(orig_shoulder_slope) or 1==1):    
                               xy_coordinate_positions["shoulder_slope"]=shoulder_slope
                               print("Reset Shoulder slope",file=sys.stderr, flush=True)
                               print(math.degrees(math.atan(xy_coordinate_positions["shoulder_slope"])),file=sys.stderr, flush=True)
                           else:
                               xy_coordinate_positions["shoulder_slope"]=orig_shoulder_slope
               
    elif( 
        (
        (abs(math.degrees(math.atan(nose_slope)))>=degrees_nose_slope_min and abs(math.degrees(math.atan(nose_slope))) <=degrees_nose_slope_max ) 
         and (xy_coordinate_positions["ear_slope"]*xy_coordinate_positions["shoulder_slope"]<=0)
         )
         or
         (
        (abs(math.degrees(math.atan(temple_slope)))>=degrees_nose_slope_min and abs(math.degrees(math.atan(temple_slope))) <=degrees_nose_slope_max ) 
         and (xy_coordinate_positions["ear_slope"]*xy_coordinate_positions["shoulder_slope"]<=0)
         )
        ):
           if (
            ((90-abs(math.degrees(math.atan(xy_coordinate_positions["temple_slope"]))))<=(90-abs(math.degrees(math.atan(xy_coordinate_positions["nose_slope"])))))
            and
            ( (math.degrees((math.atan(nose_slope))<0) 
                       and (math.degrees(math.atan(shoulder_slope))>=0 or math.degrees(math.atan(ear_slope))>=0))
                       or (math.degrees((math.atan(nose_slope))>=0) and math.degrees((math.atan(nose_slope))<=90)
                       and (math.degrees(math.atan(shoulder_slope))<0 or math.degrees(math.atan(ear_slope))<=0)
                       )
              )
            and (
            abs(math.degrees(math.atan(shoulder_slope)))>=degrees_shoulder_slope_max
            or abs(math.degrees(math.atan(ear_slope)))>=degrees_shoulder_slope_max
            )
            ):
            print ("Following Temple slope and trying to Reset shoulder slope as Temple slope is vertical and shoulder and ear are in different inclined",file=sys.stderr, flush=True)
            nose_slope=temple_slope
            xy_coordinate_positions["nose_slope"]=temple_slope
            if (math.degrees((math.atan(nose_slope))>=0) and math.degrees((math.atan(nose_slope))<=90)):
                   shoulder_slope=math.tan(math.radians(math.degrees(abs(math.atan(nose_slope)))-90))
            else:
                   shoulder_slope=math.tan(math.radians(90-math.degrees(abs(math.atan(nose_slope)))))
                   
            if (abs(shoulder_slope)<abs(orig_shoulder_slope) or 1==1):    
               xy_coordinate_positions["shoulder_slope"]=shoulder_slope
               print("Reset Shoulder slope",file=sys.stderr, flush=True)
               print(math.degrees(math.atan(xy_coordinate_positions["shoulder_slope"])),file=sys.stderr, flush=True)
            else:
               xy_coordinate_positions["shoulder_slope"]=orig_shoulder_slope
            
            
           else:
               if (
                abs(math.degrees(math.atan(shoulder_slope)))<=degrees_shoulder_slope_max+2
                or abs(math.degrees(math.atan(ear_slope)))<=degrees_shoulder_slope_max+2
                ):
                   print ("Following Shoulder slope and trying to Reset nose slope as shoulder slope is in limit and nose slope is vertical and shoulder and ear are in different inclined",file=sys.stderr, flush=True) 
                   if (math.degrees((math.atan(nose_slope))>=0) and math.degrees((math.atan(nose_slope))<=90)):
                       nose_slope=math.tan(math.radians(math.degrees(math.atan(shoulder_slope)) - 90))
                   else:
                       nose_slope=math.tan(math.radians(90+ math.degrees(math.atan(shoulder_slope))))
                   
                   if ((90-abs(math.degrees(math.atan(nose_slope))))<(90-abs(math.degrees(math.atan(orig_nose_slope))))):
                        xy_coordinate_positions["nose_slope"]=nose_slope
                        print("Reset Nose slope",file=sys.stderr, flush=True)
                        print(math.degrees(math.atan(xy_coordinate_positions['nose_slope'])),file=sys.stderr, flush=True)
                   else:
                       xy_coordinate_positions["nose_slope"]=orig_nose_slope
                   
               else:
                   print ("Following Nose slope and trying to Reset shoulder slope as nose slope is vertical and shoulder and ear are in different inclined",file=sys.stderr, flush=True)
                   if (math.degrees((math.atan(nose_slope))>=0) and math.degrees((math.atan(nose_slope))<=90)):
                        shoulder_slope=math.tan(math.radians(math.degrees(abs(math.atan(nose_slope)))-90))
                   else:
                       shoulder_slope=math.tan(math.radians(90-math.degrees(abs(math.atan(nose_slope)))))
                       
                   if (abs(shoulder_slope)<abs(orig_shoulder_slope) or 1==1):    
                       xy_coordinate_positions["shoulder_slope"]=shoulder_slope
                       print("Reset Shoulder slope",file=sys.stderr, flush=True)
                       print(math.degrees(math.atan(xy_coordinate_positions["shoulder_slope"])),file=sys.stderr, flush=True)
                   else:
                       xy_coordinate_positions["shoulder_slope"]=orig_shoulder_slope
        
        
        
        
        
    elif (((abs(math.degrees(math.atan(nose_slope)))<degrees_nose_slope_min or abs(math.degrees(math.atan(nose_slope))) >degrees_nose_slope_max )
        and (xy_coordinate_positions["shoulder_slope"]*xy_coordinate_positions["ear_slope"]>0)
        )) :
        if ( (math.degrees((math.atan(nose_slope))<0) 
                   and math.degrees(math.atan(shoulder_slope))>=0)
                   or (math.degrees((math.atan(nose_slope))>=0) and math.degrees((math.atan(nose_slope))<=90)
                   and math.degrees(math.atan(shoulder_slope))<0)
                   ):
                    print ("Following Nose slope and trying to Reset shoudler slope as shoulder slope and nose slope are out of bounds and ear and shoulder slope in same inclines and shoulder and nose are in opposite incline",file=sys.stderr, flush=True)       
                    if (math.degrees((math.atan(nose_slope))>=0) and math.degrees((math.atan(nose_slope))<=90)):
                        shoulder_slope=math.tan(math.radians(math.degrees(abs(math.atan(nose_slope)))-90))
                    else:
                        shoulder_slope=math.tan(math.radians(90-math.degrees(abs(math.atan(nose_slope)))))
                    
                    xy_coordinate_positions["shoulder_slope"]=shoulder_slope
                    
                    # if (abs(shoulder_slope)<abs(orig_shoulder_slope)):    
                        # xy_coordinate_positions["shoulder_slope"]=shoulder_slope
                        # print("Reset Shoulder slope",file=sys.stderr, flush=True)
                        # print(math.degrees(math.atan(xy_coordinate_positions["shoulder_slope"])),file=sys.stderr, flush=True)
                    # else:
                        # print ("Reverting to original shoulder slope",file=sys.stderr,flush=True)
                        # xy_coordinate_positions["shoulder_slope"]=orig_shoulder_slope
                        
        else:
           if (abs(math.degrees(math.atan(ear_slope)))<degrees_shoulder_slope_max):
               print ("Following ear slope and trying to Reset nose slope as shoulder slope and nose slope are out of bounds and ear and shoulder slope in same inclines but shoulder and nose are in same incline and ear slope in limits",file=sys.stderr, flush=True)
               shoulder_slope=ear_slope
               xy_coordinate_positions["shoulder_slope"]=ear_slope
               if (math.degrees((math.atan(nose_slope))>=0) and math.degrees((math.atan(nose_slope))<=90)):
                   nose_slope=math.tan(math.radians(math.degrees(math.atan(shoulder_slope)) - 90))
               else:
                   nose_slope=math.tan(math.radians(90+ math.degrees(math.atan(shoulder_slope))))
               
               if ((90-abs(math.degrees(math.atan(nose_slope))))<(90-abs(math.degrees(math.atan(orig_nose_slope))))):
                    xy_coordinate_positions["nose_slope"]=nose_slope
                    print("Reset Nose slope",file=sys.stderr, flush=True)
                    print(math.degrees(math.atan(xy_coordinate_positions['nose_slope'])),file=sys.stderr, flush=True)
               else:
                   xy_coordinate_positions["nose_slope"]=orig_nose_slope
          
           else:
               print ("Following Shoulder slope and trying to Reset nose slope as shoulder slope and nose slope are out of bounds and ear and shoulder slope in same inclines but shoulder and nose are in same incline and ear slope out of limits",file=sys.stderr, flush=True)
               if (math.degrees((math.atan(nose_slope))>=0) and math.degrees((math.atan(nose_slope))<=90)):
                   nose_slope=math.tan(math.radians(math.degrees(math.atan(shoulder_slope)) - 90))
               else:
                   nose_slope=math.tan(math.radians(90+ math.degrees(math.atan(shoulder_slope))))
               
               if ((90-abs(math.degrees(math.atan(nose_slope))))<(90-abs(math.degrees(math.atan(orig_nose_slope))))):
                    xy_coordinate_positions["nose_slope"]=nose_slope
                    print("Reset Nose slope",file=sys.stderr, flush=True)
                    print(math.degrees(math.atan(xy_coordinate_positions['nose_slope'])),file=sys.stderr, flush=True)
               else:
                   xy_coordinate_positions["nose_slope"]=orig_nose_slope
           
       # if (abs(shoulder_slope)<abs(orig_shoulder_slope) or 1==1):    
           # xy_coordinate_positions["shoulder_slope"]=shoulder_slope
           # print("Reset Shoulder slope",file=sys.stderr, flush=True)
           # print(math.degrees(math.atan(xy_coordinate_positions["shoulder_slope"])),file=sys.stderr, flush=True)
       # else:
           # xy_coordinate_positions["shoulder_slope"]=orig_shoulder_slope
    elif (((abs(math.degrees(math.atan(nose_slope)))<degrees_nose_slope_min or abs(math.degrees(math.atan(nose_slope))) >degrees_nose_slope_max )
        and (xy_coordinate_positions["shoulder_slope"]*xy_coordinate_positions["ear_slope"]<=0)
        )) :
        if ( (math.degrees((math.atan(nose_slope))<0) 
                   and math.degrees(math.atan(shoulder_slope))>=0)
                   or (math.degrees((math.atan(nose_slope))>=0) and math.degrees((math.atan(nose_slope))<=90)
                   and math.degrees(math.atan(shoulder_slope))<0)
                   ):
                    print ("Following Nose slope and trying to Reset shoudler slope as shoulder slope and nose slope are out of bounds and ear and shoulder slope in different inclines and shoulder and nose are different inclined",file=sys.stderr, flush=True)       
                    if (math.degrees((math.atan(nose_slope))>=0) and math.degrees((math.atan(nose_slope))<=90)):
                        shoulder_slope=math.tan(math.radians(math.degrees(abs(math.atan(nose_slope)))-90))
                    else:
                        shoulder_slope=math.tan(math.radians(90-math.degrees(abs(math.atan(nose_slope)))))
                   
                    if (abs(shoulder_slope)<abs(orig_shoulder_slope)):    
                        xy_coordinate_positions["shoulder_slope"]=shoulder_slope
                        print("Reset Shoulder slope",file=sys.stderr, flush=True)
                        print(math.degrees(math.atan(xy_coordinate_positions["shoulder_slope"])),file=sys.stderr, flush=True)
                    else:
                        print ("Reverting to original shoulder slope",file=sys.stderr,flush=True)
                        xy_coordinate_positions["shoulder_slope"]=orig_shoulder_slope
                        
        else:            
           if (abs(math.degrees(math.atan(ear_slope)))<degrees_shoulder_slope_max):
               print ("Following ear slope and trying to Reset nose slope as shoulder slope and nose slope are out of bounds and ear and shoulder slope in differemt inclines but shoulder and nose are in same incline and ear slope in limits",file=sys.stderr, flush=True)
               shoulder_slope=ear_slope
               xy_coordinate_positions["shoulder_slope"]=ear_slope
               if (math.degrees((math.atan(nose_slope))>=0) and math.degrees((math.atan(nose_slope))<=90)):
                   nose_slope=math.tan(math.radians(math.degrees(math.atan(shoulder_slope)) - 90))
               else:
                   nose_slope=math.tan(math.radians(90+ math.degrees(math.atan(shoulder_slope))))
               
               if ((90-abs(math.degrees(math.atan(nose_slope))))<(90-abs(math.degrees(math.atan(orig_nose_slope))))):
                    xy_coordinate_positions["nose_slope"]=nose_slope
                    print("Reset Nose slope",file=sys.stderr, flush=True)
                    print(math.degrees(math.atan(xy_coordinate_positions['nose_slope'])),file=sys.stderr, flush=True)
               else:
                   xy_coordinate_positions["nose_slope"]=orig_nose_slope
          
           else:
               if (abs(math.degrees((math.atan(shoulder_slope))))<(2*horizontal_offset)):
                       print ("Following Shoulder slope and trying to Reset nose slope as shoulder slope and nose slope are out of bounds and ear and shoulder slope in different inclines but shoulder and nose are in same incline and ear slope out of limits and shoulder below 2*horizontal_offset",file=sys.stderr, flush=True)
                       if (math.degrees((math.atan(nose_slope))>=0) and math.degrees((math.atan(nose_slope))<=90)):
                           nose_slope=math.tan(math.radians(math.degrees(math.atan(shoulder_slope)) - 90))
                       else:
                           nose_slope=math.tan(math.radians(90+ math.degrees(math.atan(shoulder_slope))))
                       
                       if ((90-abs(math.degrees(math.atan(nose_slope))))<(90-abs(math.degrees(math.atan(orig_nose_slope))))):
                            xy_coordinate_positions["nose_slope"]=nose_slope
                            print("Reset Nose slope",file=sys.stderr, flush=True)
                            print(math.degrees(math.atan(xy_coordinate_positions['nose_slope'])),file=sys.stderr, flush=True)
                       else:
                           xy_coordinate_positions["nose_slope"]=orig_nose_slope
               
               elif (abs(math.degrees((math.atan(nose_slope))))>(90-2*vertical_offset) 
                   and abs(math.degrees((math.atan(nose_slope))))<=90):
                    print ("Following Nose slope and trying to Reset nose slope as shoulder slope and nose slope are out of bounds and ear and shoulder slope in different inclines but shoulder and nose are in same incline and ear slope out of limits and nose above 2*vertical_offset",file=sys.stderr, flush=True)       
                   
                    if (math.degrees((math.atan(nose_slope))>=0) and math.degrees((math.atan(nose_slope))<=90)):
                        shoulder_slope=math.tan(math.radians(math.degrees(abs(math.atan(nose_slope)))-90))
                    else:
                        shoulder_slope=math.tan(math.radians(90-math.degrees(abs(math.atan(nose_slope)))))
                   
                    if (abs(shoulder_slope)<abs(orig_shoulder_slope)):    
                        xy_coordinate_positions["shoulder_slope"]=shoulder_slope
                        print("Reset Shoulder slope",file=sys.stderr, flush=True)
                        print(math.degrees(math.atan(xy_coordinate_positions["shoulder_slope"])),file=sys.stderr, flush=True)
                    else:
                        print ("Reverting to original shoulder slope",file=sys.stderr,flush=True)
                        xy_coordinate_positions["shoulder_slope"]=orig_shoulder_slope
               else:    
                   print ("Following Ear slope and trying to Reset nose slope as shoulder slope and nose slope are out of bounds and ear and shoulder slope in different inclines but shoulder and nose are in same incline and ear slope out of limits and nose below 70deg",file=sys.stderr, flush=True)
                   shoulder_slope=ear_slope
                   xy_coordinate_positions["shoulder_slope"]=ear_slope
                   if (math.degrees((math.atan(nose_slope))>=0) and math.degrees((math.atan(nose_slope))<=90)):
                       nose_slope=math.tan(math.radians(math.degrees(math.atan(shoulder_slope)) - 90))
                   else:
                       nose_slope=math.tan(math.radians(90+ math.degrees(math.atan(shoulder_slope))))
                   
                   if ((90-abs(math.degrees(math.atan(nose_slope))))<(90-abs(math.degrees(math.atan(orig_nose_slope))))):
                        xy_coordinate_positions["nose_slope"]=nose_slope
                        print("Reset Nose slope",file=sys.stderr, flush=True)
                        print(math.degrees(math.atan(xy_coordinate_positions['nose_slope'])),file=sys.stderr, flush=True)
                   else:
                       xy_coordinate_positions["nose_slope"]=orig_nose_slope
           
    else:
        print ("Following Shoulder slope as no other condition mateched across",file=sys.stderr, flush=True)
        if (math.degrees((math.atan(nose_slope))>=0) and math.degrees((math.atan(nose_slope))<=90)):
           nose_slope=math.tan(math.radians(math.degrees(math.atan(shoulder_slope)) - 90))
        else:
           nose_slope=math.tan(math.radians(90+ math.degrees(math.atan(shoulder_slope))))
        if ((90-abs(math.degrees(math.atan(nose_slope))))<(90-abs(math.degrees(math.atan(orig_nose_slope))))):
            xy_coordinate_positions["nose_slope"]=nose_slope
            print("Reset Nose slope",file=sys.stderr, flush=True)
            print(math.degrees(math.atan(xy_coordinate_positions['nose_slope'])),file=sys.stderr, flush=True)
        else:
           xy_coordinate_positions["nose_slope"]=orig_nose_slope      
    shoulder_slope=xy_coordinate_positions["shoulder_slope"]
    nose_slope=xy_coordinate_positions["nose_slope"]
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

    print ("Degrees shoulder:",math.degrees((math.atan(xy_coordinate_positions["shoulder_slope"])))," Degrees nose:",math.degrees((math.atan(xy_coordinate_positions["nose_slope"]))),file=sys.stderr, flush=True)
    if (math.degrees(abs(math.atan(xy_coordinate_positions["shoulder_slope"])))>degrees_shoulder_slope_max*6 or math.degrees(abs(math.atan(xy_coordinate_positions["nose_slope"])))<60):
        print("Error!!shoulder or nose slope needs correction- shoulder:",math.degrees(abs(math.atan(xy_coordinate_positions["shoulder_slope"]))),"nose:",math.degrees(abs(math.atan(xy_coordinate_positions["nose_slope"]))),file=sys.stderr, flush=True)
        raise Exception('shoulder or nose slope needs correction',math.degrees(abs(math.atan(xy_coordinate_positions["shoulder_slope"]))),math.degrees(abs(math.atan(xy_coordinate_positions["nose_slope"]))))
    
    #resize horizontal scale is shoulder is bent too much
    if (use_different_horizontal_vertical_scale==True and 
        ((abs(math.degrees((math.atan(xy_coordinate_positions["shoulder_slope"]))))>=1.6*degrees_shoulder_slope_max)
        or
        (abs(math.degrees((math.atan(orig_ear_slope))))>=1.4*degrees_shoulder_slope_max))
        ):
        if (
            (abs(math.degrees((math.atan(xy_coordinate_positions["shoulder_slope"]))))>=4*degrees_shoulder_slope_max)
            or
            (abs(math.degrees((math.atan(orig_ear_slope))))>=4*degrees_shoulder_slope_max)
            ):
                
                print ("Resizing Horizontal Scale as shoulder slope is more than 4 times accepted or ear more than 4 times",file=sys.stderr, flush=True)
                if (abs(math.degrees((math.atan(xy_coordinate_positions["shoulder_slope"]))))>=2*degrees_shoulder_slope_max):
                    max_horizontal_vertical_ratio=1.18
                else:
                    max_horizontal_vertical_ratio=1.08
                
        
        
        elif (
            (abs(math.degrees((math.atan(xy_coordinate_positions["shoulder_slope"]))))>=3.6*degrees_shoulder_slope_max)
            or
            (abs(math.degrees((math.atan(orig_ear_slope))))>=3.5*degrees_shoulder_slope_max)
            ):
            print ("Resizing Horizontal Scale as shoulder slope is more than 3.6 times accepted or ear more than 3.5 times",file=sys.stderr, flush=True)
            max_horizontal_vertical_ratio=1.1
            
        elif ( 
            (abs(math.degrees((math.atan(xy_coordinate_positions["shoulder_slope"]))))>=3.2*degrees_shoulder_slope_max)
            or
            (abs(math.degrees((math.atan(ear_slope))))>=3.2*degrees_shoulder_slope_max)
            ):
            print ("Resizing Horizontal Scale as shoulder slope more than 3.2 times accepted or ear more than 3.2times",file=sys.stderr, flush=True)
            max_horizontal_vertical_ratio=1.08
            
        
        elif ( 
            (abs(math.degrees((math.atan(xy_coordinate_positions["shoulder_slope"]))))>=2.8*degrees_shoulder_slope_max)
            or
            (abs(math.degrees((math.atan(ear_slope))))>=2.8*degrees_shoulder_slope_max)
            ):
            print ("Resizing Horizontal Scale as shoulder slope more than 2.8 times accepted or ear more than 2.8times",file=sys.stderr, flush=True)
            max_horizontal_vertical_ratio=1.08
            
        elif ( 
            (abs(math.degrees((math.atan(xy_coordinate_positions["shoulder_slope"]))))>=2.4*degrees_shoulder_slope_max)
            or
            (abs(math.degrees((math.atan(ear_slope))))>=2.4*degrees_shoulder_slope_max)
            ):
            print ("Resizing Horizontal Scale as shoulder slope is more than 2.4 and less than 3.6 times accepted or ear more than 2.4 times",file=sys.stderr, flush=True)
            max_horizontal_vertical_ratio=1.08
        
        elif ( 
            (abs(math.degrees((math.atan(xy_coordinate_positions["shoulder_slope"]))))>=2*degrees_shoulder_slope_max)
            or
            (abs(math.degrees((math.atan(ear_slope))))>=2*degrees_shoulder_slope_max)
            ):
            print ("Resizing Horizontal Scale as shoulder slope is more than 2 and less than 3.6 times accepted or ear more than 2 times",file=sys.stderr, flush=True)
            max_horizontal_vertical_ratio=1.08
        
        elif ( 
            (abs(math.degrees((math.atan(xy_coordinate_positions["shoulder_slope"]))))>=1.8*degrees_shoulder_slope_max)
            or
            (abs(math.degrees((math.atan(ear_slope))))>=1.8*degrees_shoulder_slope_max)
            ):
            print ("Resizing Horizontal Scale as shoulder slope is more than 1.8 and less than 2.4 times accepted or ear more than 1.8 times",file=sys.stderr, flush=True)
            max_horizontal_vertical_ratio=1.08
        else:
            print ("Resizing Horizontal Scale as shoulder slope is more than 1.6 and less than 1.8 times accepted",file=sys.stderr, flush=True)
            max_horizontal_vertical_ratio=1.08
        
        xy_coordinate_positions["vertical_reduced_circle_radius"]=round(xy_coordinate_positions["face_nose_thorax_distance"] * vertical_ratio/100)
        xy_coordinate_positions["horizontal_reduced_circle_radius"]=xy_coordinate_positions["vertical_reduced_circle_radius"] * max_horizontal_vertical_ratio
        xy_coordinate_positions["vertical_ratio"]=xy_coordinate_positions["vertical_reduced_circle_radius"]/xy_coordinate_positions["horizontal_reduced_circle_radius"]
        xy_coordinate_positions["horizontal_ratio"]=xy_coordinate_positions["horizontal_reduced_circle_radius"]/xy_coordinate_positions["vertical_reduced_circle_radius"]
        print("New Vertical Ratio:"+str(xy_coordinate_positions["vertical_ratio"]),file=sys.stderr, flush=True)
        print("New Horizontal Ratio:"+str(xy_coordinate_positions["horizontal_ratio"]),file=sys.stderr, flush=True)
        vertical_reduced_circle_radius=xy_coordinate_positions["vertical_reduced_circle_radius"]
        horizontal_reduced_circle_radius=xy_coordinate_positions["horizontal_reduced_circle_radius"]
        # xy_coordinate_positions["reduced_circle_radius"]=round(xy_coordinate_positions["face_nose_thorax_distance"] * vertical_ratio/100)
        reduced_circle_radius=xy_coordinate_positions["reduced_circle_radius"]
        xy_coordinate_positions["thorax_top_bottom_distance"]=xy_coordinate_positions["horizontal_reduced_circle_radius"]*2
        
        
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
    
    for key in positions:
      if key in interested_points:
        if isinstance(positions[key], list):
            if (positions[key][0]<min_img_x or positions[key][1]<min_img_y or positions[key][0]>max_img_x or positions[key][1]>max_img_y):
                message="Error!:"+str(key)+" is out of image bounds : ["+str(positions[key][0])+","+str(positions[key][1])+"]"
                print(message,file=sys.stderr, flush=True)
                raise Exception(message)  
    
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