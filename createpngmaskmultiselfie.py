from rembg import remove,new_session
import extractfacemaskfromimg as extract_face
from cvzone.FaceDetectionModule import FaceDetector
import mediapipe as mp
import cv2
import cvzone
import time
import os
import math as math
import numpy as np
import sys
import pprint
import PIL
from PIL import Image
from mediapipe.python._framework_bindings import image
from mediapipe.python._framework_bindings import image_frame
from mediapipe.tasks.python import vision
from mediapipe import tasks
from cvzone.ClassificationModule import Classifier
USE_CV_POSE_DETECTOR=False
pp = pprint.PrettyPrinter(indent=4)
np.set_printoptions(threshold=sys.maxsize)
model_path="./Models/selfie_multiclass_256x256.tflite"
human_path = './overlay/public3.jpg'
input_path = "./overlay/necklace8.png"
BG_COLOR = (192, 192, 192) # gray
MASK_COLOR = (255, 255, 255) # white
model="isnet-general-use"
session=new_session(model)
interested_points=["thorax_top","thorax_bottom","thorax_midpoint","right_shoulder_pivot","left_shoulder_pivot","left_shoulder","right_shoulder","nose","eye_midpoint"]
#mediapipe initialization
BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

with open(model_path, 'rb') as f:
    model_data = f.read()
base_options = BaseOptions(model_asset_buffer=model_data)
options = vision.ImageSegmenterOptions(base_options=base_options,running_mode=VisionRunningMode.IMAGE,
                                              output_category_mask=1)

#neckalce image



# Initialize the FaceDetector object
# minDetectionCon: Minimum detection confidence threshold
# modelSelection: 0 for short-range detection (2 meters), 1 for long-range detection (5 meters)
# detector = FaceDetector(minDetectionCon=0.5, modelSelection=1)

def run_histogram_equalization(img, clahe=True,y_only=False,passon=False):
    if passon==True:
        return img
    if clahe== True:
        # apply clahe
        print("Applying equilization using CLAHE",file=sys.stderr, flush=True)
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4,4))
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:,:,0])
        equalized_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    else:
        if (y_only==False):
            # equalize the histogram of the Y channel
            print("Applying YCrCB histogram correction, all YCrCb channels",file=sys.stderr, flush=True)
            # convert from RGB color-space to YCrCb
            ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
            equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
        else:
            print("Applying YCrCB histogram correction, only Y channels",file=sys.stderr, flush=True)
            # convert from RGB color-space to YCrCb
            ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb_img)
            y_eq = cv2.equalizeHist(y)
            final_img = cv2.merge((y_eq, cr, cb))
            equalized_img = cv2.cvtColor(final_img, cv2.COLOR_YCrCb2BGR)
            
    # convert back to RGB color-space from YCrCb
    
    return equalized_img



    

def detect_reapply_face_multiscale(imgOverlay,human_image_copy,segmentation_result,face_position):
    condition_hair = np.stack((segmentation_result.confidence_masks[1].numpy_view(),) * 3, axis=-1) > 0.2
    condition_background = np.stack((segmentation_result.confidence_masks[0].numpy_view(),) * 3, axis=-1) > 0.2
    condition_face_skin=np.stack((segmentation_result.confidence_masks[3].numpy_view(),) * 3, axis=-1) > 0.2
    condition_others=np.stack((segmentation_result.confidence_masks[5].numpy_view(),) * 3, axis=-1) > 0.2
    condition_bodyskin=np.stack((segmentation_result.confidence_masks[2].numpy_view(),) * 3, axis=-1) > 0.2
    neck_skin_upper_limit_y=round(face_position["thorax_top"][1]-0.25*face_position["face_nose_thorax_distance"])
    condition_bodyskin[neck_skin_upper_limit_y:,:,:]=False
    
    # print(face_position)
    # sys.exit()
    combined_condition=(condition_hair| condition_background| condition_face_skin|condition_bodyskin) 
    # combined_condition=(condition_hair| condition_background| condition_face_skin|condition_others)
    output_combined_image=np.where(combined_condition,human_image_copy, imgOverlay)
    return output_combined_image
     
     
def rotate_image(image,rotation_point, angle):
  rot_mat = cv2.getRotationMatrix2D(rotation_point, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result 
  
  
def create_mask_from_png_hsv(img):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  lower_white = np.array([0,0,168], dtype=np.uint8)
  upper_white = np.array([172,111,255], dtype=np.uint8)
  
  print(hsv.shape)
  print(img.shape)
  img_copy=cv2.cvtColor(img.copy(), cv2.COLOR_BGR2BGRA)
  
  print(img_copy.shape)
  # print(img_copy[10][10])
  bg_image = np.zeros(img_copy.shape, dtype=np.uint8)
  bg_image[:] = [192,192,192,0]
  white_pixels_mask = np.all(img_copy == [255, 255, 255,255], axis=-1)
  white_pixels_mask_rgba = np.stack([white_pixels_mask, white_pixels_mask, white_pixels_mask,white_pixels_mask], axis=2)
  
  hsv_white_mask = cv2.inRange(hsv, lower_white, upper_white)
  hsv_white_mask = cv2.bitwise_not(hsv_white_mask)
  print(hsv_white_mask.shape)
  print(hsv_white_mask[10][10])
  # output_masked_image=np.where(white_pixels_mask_rgba,bg_image,img_copy)
  
  output_masked_image=cv2.bitwise_and(img_copy, img_copy, mask = hsv_white_mask)
  cv2.imshow("PNG white removed",output_masked_image)
  return hsv_white_mask,output_masked_image
  # sys.exit()
   

def create_mask_img(img):
    '''
        Arguments:

    img - image of object in CV2 RGB format

    
    Function returns background with added object in CV2 RGB format
    
    CV2 RGB format is a np array with dimensions width x height x 3
    '''
    # img = segmentor.removeBG(img, imgBg=(255, 255, 255), cutThreshold=0.01)
    # cv2.imshow("segmented",img)
    #we first prepare the mask from img
   
    # im=remove(
    # img,
    # alpha_matting=True,
    # alpha_matting_foreground_threshold=240,
    # alpha_matting_background_threshold=10,
    # alpha_matting_erode_structure_size=1,
    # alpha_matting_base_size=100,
    # )
    
    img_copy=cv2.cvtColor(img.copy(), cv2.COLOR_BGR2BGRA)
    # cv2.imshow("img_copy",img_copy)
    bg = remove(img,
    session=session)
    img_black_bg=bg.copy()
    
    # cv2.imshow("bgremove",bg)
    img_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
   
    
    ret, mask = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.ADAPTIVE_THRESH_MEAN_C)
    print ("using Adaptive Thresholding to Mask Jewelery",file=sys.stderr, flush=True)
    # cv2.imshow("threshold mask",mask)
    # mask=cv2.threshold
    mask = cv2.bitwise_not(mask) #we need to set those regions to remove as black
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) #mask to rgb mask to get True/False
    # cv2.imshow("maskrgb",mask)
    bg_image = np.zeros(img_copy.shape, dtype=np.uint8)
    bg_image[:] = [MASK_COLOR[0],MASK_COLOR[1],MASK_COLOR[2],0]
    
    bg_image2= np.zeros(img_copy.shape, dtype=np.uint8)
    bg_image2[:] = [192,192,192,0]
    # bg_image=cv2.cvtColor( bg_image, cv2.COLOR_RGB2RGBA)
    # print(bg_image.shape)
    mask_boolean = mask[:,:,0] == 0
    
    mask_rgb_boolean = np.stack([mask_boolean, mask_boolean, mask_boolean,mask_boolean], axis=2)
    # print(mask_boolean.shape)
    # print(mask_rgb_boolean.shape)
    # print(img_copy.shape)
    # sys.exit()
    output_masked_image=np.where(mask_rgb_boolean,img_copy, bg_image)
    # cv2.imshow("without white filter",output_masked_image)
    # print(output_masked_image.shape)
    output_masked_image2=cv2.cvtColor(output_masked_image,cv2.COLOR_BGRA2BGR)
    lower_white = np.array([240,240,240,0], dtype=np.uint8)
    upper_white = np.array([255,255,255,255], dtype=np.uint8)
    white_pixels_mask = cv2.inRange(output_masked_image,lower_white,upper_white)
   
    white_pixels_mask_rgba = np.stack([white_pixels_mask, white_pixels_mask, white_pixels_mask,white_pixels_mask], axis=2)
    output_masked_image=np.where(white_pixels_mask_rgba,bg_image,img_copy)

    output_masked_image2=np.where(white_pixels_mask_rgba,bg_image2,img_copy)
    # cv2.imshow("mask with white filter",output_masked_image2)
    # sys.exit()
    
    
    return mask,output_masked_image



def rescale_coordinates(positions,scale):
    rescaled_positions={}
    for key in positions:
        # print(type(positions[key]))
        # print(positions[key])
        if isinstance(positions[key], list):
            rescaled_positions[key]=[round(positions[key][0]*scale),round(positions[key][1]*scale)]
        else:
            rescaled_positions[key]=round(positions[key]*scale)
    # print(real_positions)    
    return rescaled_positions 
    


def xy_coordinate_positions(positions):
    # print("xy_coordinate_positions")
    real_positions = {}
    for key in positions:
        # print(type(positions[key]))
        # print(positions[key])
        if isinstance(positions[key], list):
            real_positions[key]=[positions[key][0],-1*positions[key][1]]
        else:
            real_positions[key]=positions[key]
    # print(real_positions)    
    return real_positions    

def img_position_from_xy_coordinate_positions(positions):
    # print("img_position_from_xy_coordinate_positions")
    img_positions = {}
    for key in positions:
        # print(type(positions[key]))
        # print(positions[key])
        if isinstance(positions[key], list):
            img_positions[key]=[positions[key][0],-1*positions[key][1]]
        else:
            img_positions[key]=positions[key]
        
    return img_positions
    
def offset_coordinates(positions,x,y):
    # print("img_position_from_xy_coordinate_positions")
    offset_positions = {}
    for key in positions:
        # print(type(positions[key]))
        # print(positions[key])
        if isinstance(positions[key], list):
            offset_positions[key]=[positions[key][0]-x,positions[key][1]-y]
        
        
    return offset_positions
    
def slope_intercept(p1,p2):
    # print(p1,p2)
    slope=(p2[1]-p1[1])/(p2[0]-p1[0])
    # print(math.degrees(math.atan(slope)))
    intercept=p1[1]-slope*p1[0]
    return slope,intercept

def get_jewellery_image_mask(img):
    if (img.shape[2]==4):
        if (np.any(img[:,:,3]==0)):
            print("Jewellery Image has Alpha - using it directly as mask",file=sys.stderr, flush=True)
            masked_image=img
        else:
            print("Jewellery Image has Alpha - but the mask has no 0 alpha values",file=sys.stderr, flush=True)
            mask,masked_image=create_mask_img(img)
    else:
        mask,masked_image=create_mask_img(img)
    return (masked_image)


def get_selfie_human_image(human_image,RUN_CV_SELFIE_SEGMENTER=True,use_different_horizontal_vertical_scale=False,force_shoulder_z_alignment=False,use_cv_pose_detector=True):
    human_image,face_position=extract_face.getSelfieImageandFaceLandMarkPoints(human_image,RUN_CV_SELFIE_SEGMENTER,use_different_horizontal_vertical_scale,force_shoulder_z_alignment,use_cv_pose_detector)
    human_image_tf = mp.Image(image_format=mp.ImageFormat.SRGB, data=human_image)
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        segmentation_result = segmenter.segment(human_image_tf)
    human_image=human_image_tf.numpy_view().copy()
    return (human_image,face_position,segmentation_result)
    
    
def get_jewellery_perspective_image(img,jewellery_position,face_position,debug=False):
    
    masked_image=get_jewellery_image_mask(img)
    jewellery_position["right_shoulder_pivot"]=[0,0]
    jewellery_position["left_shoulder_pivot"]=[0,0]
    jewellery_position["thorax_midpoint"]=[0,0]
    
    
    
    jewellery_xy_position={}
    jewellery_xy_position=xy_coordinate_positions(jewellery_position)
   
    jewellery_xy_position["thorax_midpoint"]=[round((jewellery_xy_position["thorax_top"][0]+jewellery_xy_position["thorax_bottom"][0])/2),round((jewellery_xy_position["thorax_top"][1]+jewellery_xy_position["thorax_bottom"][1])/2)]
    jewellery_position['thorax_top_bottom_distance']=math.dist(jewellery_xy_position['thorax_top'],jewellery_xy_position['thorax_bottom'])
    jewellery_xy_position['right_shoulder_pivot'][0]=round((jewellery_xy_position["thorax_midpoint"][0]-jewellery_position['thorax_top_bottom_distance']/2))
    jewellery_xy_position['right_shoulder_pivot'][1]=jewellery_xy_position["thorax_midpoint"][1]
    jewellery_xy_position['left_shoulder_pivot'][0]=round((jewellery_xy_position["thorax_midpoint"][0]+jewellery_position['thorax_top_bottom_distance']/2))
    jewellery_xy_position['left_shoulder_pivot'][1]=jewellery_xy_position["thorax_midpoint"][1]

    jewellery_position['right_left_shoulder_pivot_distance']=math.dist(jewellery_xy_position['right_shoulder_pivot'],jewellery_xy_position['left_shoulder_pivot'])
    jewellery_xy_position['thorax_top_bottom_distance']=jewellery_position['thorax_top_bottom_distance']
    jewellery_xy_position['right_left_shoulder_pivot_distance']=jewellery_position['right_left_shoulder_pivot_distance']
    jewellery_position=img_position_from_xy_coordinate_positions(jewellery_xy_position)
    # print("----Jewellery Position----")
    # print(jewellery_position)

    face_xy_position=xy_coordinate_positions(face_position)
    face_position=img_position_from_xy_coordinate_positions(face_xy_position)

    # print("-----Face Coordinates-----")
    # print(face_position)
    # cv2.imshow("Masked Image",human_image)

    jewellery_resize_scale=face_position["thorax_top_bottom_distance"]/jewellery_position["thorax_top_bottom_distance"]

    # print("-----Jewellery Resize Scale-----")
    # print(jewellery_resize_scale)

    jewellery_position["image_width"] = masked_image.shape[1]
    jewellery_position["image_height"] = masked_image.shape[0]
    jewellery_xy_position["image_width"]=jewellery_position["image_width"]
    jewellery_xy_position["image_height"]=jewellery_position["image_height"]
    jewellery_position=rescale_coordinates(jewellery_position,jewellery_resize_scale)
    jewellery_xy_position=rescale_coordinates(jewellery_xy_position,jewellery_resize_scale)
    # cv2.imshow("masked_image",masked_image)
    masked_image=cv2.resize(masked_image, (jewellery_position["image_width"],jewellery_position["image_height"]), interpolation = cv2.INTER_LANCZOS4)
    # cv2.imshow("resized image opencv",masked_image)

    # PIL_masked_image= Image.fromarray(cv2.cvtColor(masked_image, cv2.COLOR_BGRA2RGBA))
    # PIL_masked_image=PIL_masked_image.resize((jewellery_position["image_width"],jewellery_position["image_height"]),Image.Resampling.LANCZOS )
    # masked_image=cv2.cvtColor(np.array(PIL_masked_image), cv2.COLOR_RGBA2BGRA)
    # cv2.imshow("resized masked_image_pillow",masked_image)

    if (debug==True):
        print("in get_jewellery_perspective_image")
        print(debug)
        for key in jewellery_position:
            if isinstance(jewellery_position[key], list):
            # print(key)
                cv2.circle(masked_image, (jewellery_position[key][0],jewellery_position[key][1]), radius=3, color=(0, 255, 255), thickness=-1)

        cv2.circle(masked_image,jewellery_position["thorax_midpoint"],radius=round(jewellery_position['thorax_top_bottom_distance']/2),color=(0,0,255),thickness=1)

    # cv2.imshow("mask",masked_image)
    # print("-----Jewellery Scaled Coordinates-----")
    # print(jewellery_position)
    jewellery_to_human_image_midpoint_offset=[0,0]
    jewellery_to_human_image_midpoint_offset[0]=face_position["thorax_midpoint"][0]-jewellery_position["thorax_midpoint"][0]
    jewellery_to_human_image_midpoint_offset[1]=face_position["thorax_midpoint"][1]-jewellery_position["thorax_midpoint"][1]
    
    jewellery_transform_final_points=offset_coordinates(face_position,jewellery_to_human_image_midpoint_offset[0],jewellery_to_human_image_midpoint_offset[1])

    # print("-----Jewellery Transform Coordinates-----")
    # print(jewellery_transform_final_points)

    input_pts=np.float32([jewellery_position["left_shoulder_pivot"],jewellery_position["thorax_top"],jewellery_position["right_shoulder_pivot"],jewellery_position["thorax_bottom"]])
    output_pts=np.float32([jewellery_transform_final_points["left_shoulder_pivot"],jewellery_transform_final_points["thorax_top"],jewellery_transform_final_points["right_shoulder_pivot"],jewellery_transform_final_points["thorax_bottom"]])

    M = cv2.getPerspectiveTransform(input_pts,output_pts)

    important_points=np.float32([
    [[jewellery_position["thorax_midpoint"][0],jewellery_position["thorax_midpoint"][1]]],
    [[0,0]],[[masked_image.shape[1],0]],
    [[masked_image.shape[1],masked_image.shape[0]]],[[0,masked_image.shape[0]]],
    ])

    transformed_important_points=cv2.perspectiveTransform(important_points,M)
    # print("transformed points")
    # print(important_points)
    # print("<==>")
    # print(transformed_important_points)

    yadd=0
    xadd=0
    for points in transformed_important_points:
        # print(points[0][0])
        if (points[0][0]<xadd):
            xadd=points[0][0]
        if (points[0][1]<yadd):
            yadd=points[0][1]

    print(xadd,yadd)
    xadd=round(abs(xadd))
    yadd=round(abs(yadd))
    perspective_masked_image=cv2.warpPerspective(masked_image,M,(masked_image.shape[1]+xadd, masked_image.shape[0]+yadd),flags=cv2.INTER_LINEAR)
    
    # print ("jewellery position",file=sys.stderr, flush=True)
    # print(jewellery_position,file=sys.stderr, flush=True)
    
    # print ("jewellery xy position",file=sys.stderr, flush=True)
    # print(jewellery_xy_position,file=sys.stderr, flush=True)
    
    return (perspective_masked_image,masked_image,jewellery_position,face_position)



def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    # alpha_channel=np.where((foreground[:, :, 3]>15)  ,1.0 ,0.0)
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0
    # print(alpha_channel.shape)
    # alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0
    
    alpha_channel=np.where((alpha_channel>0.05) & (alpha_channel<0.4),alpha_channel+0.1,alpha_channel)
    alpha_channel[alpha_channel<0.1]=0
    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))
    # cv2.imshow("Alpha mask",alpha_mask)
    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite
    return background



def overlay_jewellery_on_face(jewellery_position,face_position,human_image,perspective_masked_image,segmentation_result):
    human_image_copy=human_image.copy()
    overlaypoint_x=face_position["thorax_midpoint"][0]-jewellery_position["thorax_midpoint"][0]
    overlaypoint_y=face_position["thorax_midpoint"][1]-jewellery_position["thorax_midpoint"][1]
    # print ("Overlaying")
    # print(face_position)
    # print(jewellery_position)
    # print(overlaypoint_x, overlaypoint_y)

    # imgOverlay = cvzone.overlayPNG(human_image, perspective_masked_image, pos=[overlaypoint_x, overlaypoint_y])
    imgOverlay=add_transparent_image(human_image,perspective_masked_image,overlaypoint_x,overlaypoint_y)
    # cv2.imshow("IO",imgOverlay)
    # cv2.imshow("Perspective",perspective_masked_image)

    imgOverlay=detect_reapply_face_multiscale(imgOverlay,human_image_copy,segmentation_result,face_position)
    return imgOverlay
    
    
    
    
def get_sample_preview_image(jewellery_image,jewellery_position,human_image,RUN_CV_SELFIE_SEGMENTER=True,use_different_horizontal_vertical_scale=False,force_shoulder_z_alignment=False,use_cv_pose_detector=True):
    try:
        human_image,face_position,segmentation_result=get_selfie_human_image(human_image,RUN_CV_SELFIE_SEGMENTER,use_different_horizontal_vertical_scale,force_shoulder_z_alignment,use_cv_pose_detector)
        human_image_copy=human_image.copy()
    except:
        raise Exception("Please ensure your shoulders are horizontal and your face is vertical and the upper thorax and face is facing the camera with proper lighting")
        
    try:    
        perspective_masked_image,masked_image,jewellery_position,face_position=get_jewellery_perspective_image(jewellery_image,jewellery_position,face_position,debug=True)
        horizontal_reduced_circle_radius=face_position["horizontal_reduced_circle_radius"]
        vertical_reduced_circle_radius=face_position["vertical_reduced_circle_radius"]
        for key in face_position:
             if isinstance(face_position[key], list):
              if key in interested_points:
                # print(key)
                cv2.circle(human_image, (face_position[key][0],face_position[key][1]), radius=3, color=(0, 0, 0), thickness=-1)
        if use_different_horizontal_vertical_scale==True:
                center=(int(face_position["thorax_midpoint"][0]),int(face_position["thorax_midpoint"][1]))
                axes=(int(face_position["horizontal_reduced_circle_radius"]),int(face_position["vertical_reduced_circle_radius"]))
                print ("Elliptical Markings Angle",file=sys.stderr, flush=True)
                print(-1*math.degrees(face_position["shoulder_slope"]),file=sys.stderr, flush=True)
                cv2.ellipse(human_image,center,axes,-1*math.degrees(face_position["shoulder_slope"]),0,360,(0,0,255),1)
        else:
            cv2.circle(human_image,face_position["thorax_midpoint"],radius=horizontal_reduced_circle_radius,color=(255,0,0),thickness=1)
        
        cv2.circle(perspective_masked_image,(jewellery_position["thorax_midpoint"][0],jewellery_position["thorax_midpoint"][1]),5,color=(0,255,255),thickness=-1)

        imgOverlay=overlay_jewellery_on_face(jewellery_position,face_position,human_image,perspective_masked_image,segmentation_result)
        # final_image=run_histogram_equalization(imgOverlay)
        return imgOverlay
    
    except:
        raise Exception("Unable to determine Jewellery points")
    
    

def get_final_image(jewellery_image,jewellery_position, human_image,RUN_CV_SELFIE_SEGMENTER=True,debug=False,use_different_horizontal_vertical_scale=False,force_shoulder_z_alignment=False,use_cv_pose_detector=True):
    try:
        human_image,face_position,segmentation_result=get_selfie_human_image(human_image,RUN_CV_SELFIE_SEGMENTER,use_different_horizontal_vertical_scale,force_shoulder_z_alignment,use_cv_pose_detector)
        human_image_copy=human_image.copy()
    except:
        raise Exception("Please ensure your shoulders are horizontal and your face is vertical and the upper thorax and face is facing the camera with proper lighting")
        
    
    try:
        perspective_masked_image,masked_image,jewellery_position,face_position=get_jewellery_perspective_image(jewellery_image,jewellery_position,face_position,debug)
        horizontal_reduced_circle_radius=face_position["horizontal_reduced_circle_radius"]
        vertical_reduced_circle_radius=face_position["vertical_reduced_circle_radius"]
        if (debug==True):
            for key in face_position:
                 if isinstance(face_position[key], list):
                  if key in interested_points:
                    # print(key)
                    cv2.circle(human_image, (face_position[key][0],face_position[key][1]), radius=3, color=(0, 0, 0), thickness=-1)

            if use_different_horizontal_vertical_scale==True:
                center=(int(face_position["thorax_midpoint"][0]),int(face_position["thorax_midpoint"][1]))
                axes=(int(face_position["horizontal_reduced_circle_radius"]),int(face_position["vertical_reduced_circle_radius"]))
                print ("Elliptical Markings Angle",file=sys.stderr, flush=True)
                print(-1*math.degrees(face_position["shoulder_slope"]),file=sys.stderr, flush=True)
                cv2.ellipse(human_image,center,axes,-1*math.degrees(face_position["shoulder_slope"]),0,360,(0,0,255),1)
            else:
                cv2.circle(human_image,face_position["thorax_midpoint"],radius=horizontal_reduced_circle_radius,color=(255,0,0),thickness=1)
            
            cv2.circle(perspective_masked_image,(jewellery_position["thorax_midpoint"][0],jewellery_position["thorax_midpoint"][1]),5,color=(0,255,255),thickness=-1)

            
        imgOverlay=overlay_jewellery_on_face(jewellery_position,face_position,human_image,perspective_masked_image,segmentation_result)
        # final_image=run_histogram_equalization(imgOverlay)
        return imgOverlay
    except:
        raise Exception("Unable to determine Jewellery points")
    
def main():
    img = cv2.imread(input_path,cv2.IMREAD_UNCHANGED)
    
    # cv2.imshow("neckalce",img)
    #human image
    human_image=cv2.imread(human_path,cv2.IMREAD_UNCHANGED)
    RUN_CV_SELFIE_SEGMENTER=True
    #initialization of mediapipe selfie_multiclass
    # cv2.imshow("selfie",human_image)
    
    human_image,face_position,segmentation_result=get_selfie_human_image(human_image,RUN_CV_SELFIE_SEGMENTER)
    cv2.namedWindow("Masked Image")
    cv2.moveWindow("Masked Image", 10,10)
    human_image_copy=human_image.copy()
    

    # print(masked_image.shape)
    # print(masked_image[0][0])




    #necklace1.jpg
    # jewellery_position={
    # 'thorax_top':[404,320],
    # 'thorax_bottom':[404,740],
    # 'thorax_midpoint':[0,0],
    # 'left_shoulder_pivot':[794,456],
    # 'right_shoulder_pivot':[5,456]
    # }

    # # #necklace2.jpg
    # jewellery_position={
    # 'thorax_top':[184,165],
    # 'thorax_bottom':[184,403],
    # 'thorax_midpoint':[0,0],
    # 'left_shoulder_pivot':[306,304],
    # 'right_shoulder_pivot':[84,304]
    # }


    # #necklace3.jpg
    # jewellery_position={
    # 'thorax_top':[203,307],
    # 'thorax_bottom':[203,481],
    # 'thorax_midpoint':[0,0],
    # 'left_shoulder_pivot':[385,392],
    # 'right_shoulder_pivot':[25,392]
    # }


    # ##necklace4.jpg
    # jewellery_position={
    # 'thorax_top':[225,298],
    # 'thorax_bottom':[225,525],
    # 'thorax_midpoint':[0,0],
    # 'left_shoulder_pivot':[385,392],
    # 'right_shoulder_pivot':[25,392]
    # }


    # ##necklace5.jpg
    # jewellery_position={
    # 'thorax_top':[245,160],
    # 'thorax_bottom':[245,409],
    # 'thorax_midpoint':[0,0],
    # 'left_shoulder_pivot':[385,392],
    # 'right_shoulder_pivot':[25,392]
    # }


    # ##necklace6.jpg
    # jewellery_position={
    # 'thorax_top':[111,40],
    # 'thorax_bottom':[111,240],
    # 'thorax_midpoint':[0,0],
    # 'left_shoulder_pivot':[385,392],
    # 'right_shoulder_pivot':[25,392]
    # }

    # ##necklace7.png
    # jewellery_position={
    # 'thorax_top':[128,93],
    # 'thorax_bottom':[128,293],
    # 'thorax_midpoint':[0,0],
    # 'left_shoulder_pivot':[385,392],
    # 'right_shoulder_pivot':[25,392]
    # }

    ##necklace8.png
    jewellery_position={
    'thorax_top':[180,90],
    'thorax_bottom':[180,275],
    'thorax_midpoint':[0,0],
    'left_shoulder_pivot':[385,392],
    'right_shoulder_pivot':[25,392]
    }

    # ##necklace9.png
    # jewellery_position={
    # 'thorax_top':[417,257],
    # 'thorax_bottom':[417,757],
    # 'thorax_midpoint':[0,0],
    # 'left_shoulder_pivot':[385,392],
    # 'right_shoulder_pivot':[25,392]
    # }

    # ##necklace10.png
    # jewellery_position={
    # 'thorax_top':[143,111],
    # 'thorax_bottom':[143,301],
    # 'thorax_midpoint':[0,0],
    # 'left_shoulder_pivot':[385,392],
    # 'right_shoulder_pivot':[25,392]
    # }


    
    perspective_masked_image,masked_image,jewellery_position,face_position=get_jewellery_perspective_image(img,jewellery_position,face_position,debug=True)
    cv2.imshow("Perspective",perspective_masked_image)
    horizontal_reduced_circle_radius=face_position["horizontal_reduced_circle_radius"]
    vertical_reduced_circle_radius=face_position["vertical_reduced_circle_radius"]
    for key in face_position:
         if isinstance(face_position[key], list):
            # print(key)
            cv2.circle(human_image, (face_position[key][0],face_position[key][1]), radius=3, color=(0, 0, 0), thickness=-1)

    cv2.circle(human_image,face_position["thorax_midpoint"],radius=horizontal_reduced_circle_radius,color=(255,0,0),thickness=1)
    cv2.circle(human_image,face_position["thorax_midpoint"],radius=vertical_reduced_circle_radius,color=(0,0,255),thickness=1)

    cv2.circle(perspective_masked_image,(jewellery_position["thorax_midpoint"][0],jewellery_position["thorax_midpoint"][1]),5,color=(0,255,255),thickness=-1)

    imgOverlay=overlay_jewellery_on_face(jewellery_position,face_position,human_image,perspective_masked_image_no_points,segmentation_result)
    # final_image=run_histogram_equalization(imgOverlay)
    final_image=imgOverlay
    cv2.imshow("Masked Image",final_image)
    ### use cv2.imencode to encode in format for rest api
    # cv2.imwrite("D:\\VTON\\overlay\\final_image.jpg",imgOverlay)

    # final_image[:,:,:]=

    # imgOut = segmentor.removeBG(imgOverlay, imgBg=(255, 255, 255), cutThreshold=0.4)

    # cv2.moveWindow("Masked Image",10,10)




if __name__ == "__main__":
    main()