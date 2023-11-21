import mediapipe as mp
import cv2
import math
from mediapipe.tasks import python
import sys
from mediapipe.tasks.python import vision
import pprint as pp
import sys
model_path = 'D://VTON//Models//pose_landmarker_heavy.task'

human_path="D://VTON//overlay//human_image34.jpg"
POSEDETECTOR_BODY_PARTS=["nose","left eye (inner)","left eye","left eye (outer)","right eye (inner)",
"right eye","right eye (outer)","left ear","right ear","mouth (left)","mouth (right)",
"left shoulder","right shoulder","left elbow","right elbow","left wrist","right wrist",
"left pinky","right pinky","left index","right index","left thumb","right thumb",
"left hip","right hip","left knee","right knee","left ankle","right ankle",
"left heel","right heel","left foot index","right foot index"]

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
human_image=cv2.imread(human_path,cv2.IMREAD_UNCHANGED)
# Load the input image from an image file.
mp_image = mp.Image.create_from_file(human_path)
image_height, image_width, _ = human_image.shape
# Load the input image from a numpy array.
# mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

mp_pose_landmark_list={}
with PoseLandmarker.create_from_options(options) as landmarker:
    pose_landmarker_result = landmarker.detect(mp_image)
    # pp.pprint(pose_landmarker_result.pose_landmarks[0][0].x)
    for index, elem in enumerate(POSEDETECTOR_BODY_PARTS):
        x=round(pose_landmarker_result.pose_landmarks[0][index].x*image_width)
        y=round(pose_landmarker_result.pose_landmarks[0][index].y*image_height)
        z=pose_landmarker_result.pose_landmarks[0][index].z
        print(elem,":",x,"--",y,"--",z)
        mp_pose_landmark_list[elem]=[]
        mp_pose_landmark_list[elem]=[x,y,z]
    print ("Normalized Shoulder Distance")
    print (mp_pose_landmark_list["left shoulder"][2]-mp_pose_landmark_list["right shoulder"][2])