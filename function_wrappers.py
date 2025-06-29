import createpngmaskmultiselfie as overlay
from flask import Flask, request, make_response, render_template,jsonify
from flask_cors import CORS, cross_origin
import base64
import numpy as np
import cv2
from waitress import serve
import time
import sys
app = Flask(__name__)
CORS(app)


def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response

def resizeAndPad(img, size, padColor=255):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA

    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = float(w)/h 
    saspect = float(sw)/sh

    if (saspect > aspect) or ((saspect == 1) and (aspect <= 1)):  # new horizontal image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = float(sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
        
    elif (saspect < aspect) or ((saspect == 1) and (aspect >= 1)):  # new vertical image
        new_w = sw
        new_h = np.round(float(new_w) / aspect).astype(int)
        pad_vert = float(sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0

    # set pad color
    if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img
    
    




def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    # old (python 2 version):
    # nparr = np.fromstring(encoded_data.decode('base64'), np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    return img





def get_masked_image(jewellery_image,jewellery_position, human_image,RUN_CV_SELFIE_SEGMENTER=False,debug=True,use_different_horizontal_vertical_scale=False,force_shoulder_z_alignment=False,use_cv_pose_detector=True):
    imgOut=overlay.get_final_image(jewellery_image,jewellery_position, human_image,RUN_CV_SELFIE_SEGMENTER,debug,use_different_horizontal_vertical_scale,force_shoulder_z_alignment,use_cv_pose_detector)
    return imgOut


@app.route('/uploadfile', methods=['GET'])
def uploadfile():
    return render_template("uploadimage.html")

@app.route('/', methods=['GET'])
def index():
    return 'Flask Webserver for Serving VTON'

@app.route('/overlayimage', methods=['GET','POST','OPTIONS'])
@cross_origin()
def overlayimage():
    status="success"
    if (request.method=="OPTIONS"):
        return _build_cors_preflight_response()
    
    if (request.method=="POST"):
        content = request.json
        try:
            print("Points",file=sys.stderr, flush=True)
            print(content['points'],file=sys.stderr, flush=True)
            jewellery_image= data_uri_to_cv2_img(content["jewellery_image"])
            human_image=data_uri_to_cv2_img(content["human_image"])
            jewellery_position={
                'thorax_top':[float(content['points']['thorax_top_x']),float(content['points']['thorax_top_y'])],
                'thorax_bottom':[float(content['points']['thorax_bottom_x']),float(content['points']['thorax_bottom_y'])],
                }
            print("we have a human image",file=sys.stderr, flush=True) 
            
        except:
            message="Unable to extract data from input"
            status="error"
            image=""
        
        
    else:       
        #copy paste values from list of jewellery_position values given above for relvant image
        ## For exmaple, if using necklace11, use thorax_top and thorax_bottom from necklace7 above
        #necklace11
        jewellery_position={
        'thorax_top':[180,150],
        'thorax_bottom':[180,430],
        }
        jewellery_image=cv2.imread("./overlay/necklace11.png",cv2.IMREAD_UNCHANGED)
        human_image=cv2.imread("./overlay/human_image21.jpg",cv2.IMREAD_UNCHANGED)
    if (status=="success"):    
        if (human_image.shape[2]==4):
            print("Converting PNG to BGR")
            human_image=cv2.cvtColor(human_image, cv2.COLOR_BGRA2BGR)
            
        
            
        print(time.time(),file=sys.stderr, flush=True)
        print ("Jewellery Image dimensions:",file=sys.stderr, flush=True)
        print(jewellery_image.shape,file=sys.stderr, flush=True)
        print ("Human Image dimensions:",file=sys.stderr, flush=True)
        print(human_image.shape,file=sys.stderr, flush=True)
        
        
        if ((int(jewellery_position["thorax_top"][0])>0 and int(jewellery_position["thorax_top"][0])<jewellery_image.shape[1])
            and 
            (int(jewellery_position["thorax_top"][1])>0 and int(jewellery_position["thorax_top"][1])<jewellery_image.shape[0])
            and 
            (int(jewellery_position["thorax_bottom"][0])>0 and int(jewellery_position["thorax_bottom"][0])<jewellery_image.shape[1])
            and 
            (int(jewellery_position["thorax_bottom"][1])>0 and int(jewellery_position["thorax_bottom"][1])<jewellery_image.shape[0])):
        
            status="success"
            message=""
        else:
            status="error"
            message="Invalid Coordinates for Jewellery"
            image=human_image
            
    
    
    if (status=="success"):
        try:
            imgOut=get_masked_image(jewellery_image,jewellery_position,human_image,RUN_CV_SELFIE_SEGMENTER=False,debug=False,use_different_horizontal_vertical_scale=True,force_shoulder_z_alignment=True,use_cv_pose_detector=False)
            if (content.get("resize","true")=="true"):
                print("resizing true selected",file=sys.stderr, flush=True)
                if (imgOut.shape[0]>400 and imgOut.shape[1]>400):
                    imgOut=resizeAndPad(imgOut,(400,400))
                    print("resizing Output Image as both >400",file=sys.stderr, flush=True)
                elif (imgOut.shape[0]>600 or imgOut.shape[1]>600):
                    imgOut=resizeAndPad(imgOut,(500,500))
                    print("resizing Output as one >600",file=sys.stderr, flush=True)
        except Exception as e:
            status="error"
            message=str(e)
            image=human_image
        
    print(time.time(),file=sys.stderr, flush=True)
    # image_url = request.args.get('imageurl')
    # requested_url = urllib.urlopen(image_url)
    # image_array = np.asarray(bytearray(requested_url.read()), dtype=np.uint8)
    # img = cv2.imdecode(image_array, -1)
    # Do some processing, get output_img
    
    if (status=="success"):
        retval, buffer = cv2.imencode('.png', imgOut)
    if (request.method=="POST"):
        data = { 
                "status" : status, 
                "message" : message,
            }
            
        if (status=="success"):
            buffer_b64encoded = base64.b64encode(buffer)
            data["image"]=buffer_b64encoded.decode('utf-8')
        if (status=="error"):
            print ("Error:",file=sys.stderr, flush=True)
            print (data,file=sys.stderr, flush=True)
        return jsonify(data)
    else:
        response=make_response(buffer.tobytes())
        response.headers['Content-Type'] = 'image/png'
        buffer_bytes=buffer.tobytes()
        return response
    
    




@app.route('/preview', methods=['GET','POST','OPTIONS'])
@cross_origin()
def preview():
    
    status="success"
    if (request.method=="OPTIONS"):
        return
    
    if (request.method=="POST"):
        content = request.json
        try:
            
            print("Points",file=sys.stderr, flush=True)
            print(content['points'],file=sys.stderr, flush=True)
            jewellery_image= data_uri_to_cv2_img(content["jewellery_image"])
            jewellery_position={
                'thorax_top':[float(content['points']['thorax_top_x']),float(content['points']['thorax_top_y'])],
                'thorax_bottom':[float(content['points']['thorax_bottom_x']),float(content['points']['thorax_bottom_y'])],
                }
        except:
            message="Unable to extract data from input"
            status="error"
            image=""
            
    else:
        #necklace11.png
        jewellery_position={
            'thorax_top':[180,150],
            'thorax_bottom':[180,430],
            }
        jewellery_image=cv2.imread("./overlay/necklace11.png",cv2.IMREAD_UNCHANGED)
    
    
    if (status=="success"):
    
        human_image=cv2.imread('./overlay/human_image20.jpg',cv2.IMREAD_UNCHANGED)
        
            
        print(time.time(),file=sys.stderr, flush=True)
        print ("Jewellery Image dimensions:",file=sys.stderr, flush=True)
        print(jewellery_image.shape,file=sys.stderr, flush=True)
        print ("Human Image dimensions:",file=sys.stderr, flush=True)
        print(human_image.shape,file=sys.stderr, flush=True)
        
        if ((int(jewellery_position["thorax_top"][0])>0 and int(jewellery_position["thorax_top"][0])<jewellery_image.shape[1])
            and 
            (int(jewellery_position["thorax_top"][1])>0 and int(jewellery_position["thorax_top"][1])<jewellery_image.shape[0])
            and 
            (int(jewellery_position["thorax_bottom"][0])>0 and int(jewellery_position["thorax_bottom"][0])<jewellery_image.shape[1])
            and 
            (int(jewellery_position["thorax_bottom"][1])>0 and int(jewellery_position["thorax_bottom"][1])<jewellery_image.shape[0])):
            
            status="success"
            message=""
        else:
            status="error"
            message="Invalid Coordinates for Jewellery"
            image=jewellery_image
        
    
    if (status=="success"):
        try:
            imgOut=overlay.get_sample_preview_image(jewellery_image,jewellery_position,human_image,RUN_CV_SELFIE_SEGMENTER=False,use_different_horizontal_vertical_scale=True,force_shoulder_z_alignment=True,use_cv_pose_detector=False)
            print("content")
            print(content,file=sys.stderr, flush=True)
            print(content.get("resize","true"),file=sys.stderr, flush=True)
            if (content.get("resize","true")=="true"):
                print("resizing true passed",file=sys.stderr, flush=True)
                if (imgOut.shape[0]>400 and imgOut.shape[1]>400):
                    imgOut=resizeAndPad(imgOut,(400,400))
                    print("resizing Output Image as both >400",file=sys.stderr, flush=True)
                elif (imgOut.shape[0]>600 or imgOut.shape[1]>600):
                    imgOut=resizeAndPad(imgOut,(500,500))
                    print("resizing Output as one >600",file=sys.stderr, flush=True)
        
        except Exception as e:
            status="error"
            message=str(e)
            image=jewellery_image
            
    print(time.time(),file=sys.stderr, flush=True)
    # imgOut=get_masked_image(jewellery_image,jewellery_position,human_image,RUN_CV_SELFIE_SEGMENTER=True,debug=False)
    # image_url = request.args.get('imageurl')
    # requested_url = urllib.urlopen(image_url)
    # image_array = np.asarray(bytearray(requested_url.read()), dtype=np.uint8)
    # img = cv2.imdecode(image_array, -1)
    # Do some processing, get output_img
    if (status=="success"):
        retval, buffer = cv2.imencode('.png', imgOut)
    if (request.method=="POST"):
        data = { 
                "status" : status, 
                "message" : message,
            }
            
        if (status=="success"):
            buffer_b64encoded = base64.b64encode(buffer)
            data["image"]=buffer_b64encoded.decode('utf-8')
        if (status=="error"):
            print ("Error:",file=sys.stderr, flush=True)
            print (data,file=sys.stderr, flush=True)
        return jsonify(data)
    else:
        retval, buffer = cv2.imencode('.png', imgOut)
        response=make_response(buffer.tobytes())
        response.headers['Content-Type'] = 'image/png'
        buffer_bytes=buffer.tobytes()
        return response


if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=8080,threads=10)






