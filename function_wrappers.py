import createpngmaskmultiselfie as overlay
from flask import Flask, request, make_response, render_template,redirect,jsonify
from flask_cors import CORS, cross_origin
import json
import base64
from codecs import encode
import numpy as np
import cv2
from waitress import serve
import time
import sys
import urllib
app = Flask(__name__)
CORS(app)


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
    # 'thorax_top':[210,307],
    # 'thorax_bottom':[210,481],
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
    # 'thorax_top':[245,120],
    # 'thorax_bottom':[245,369],
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

    # ##necklace8.png
    # jewellery_position={
    # 'thorax_top':[170,90],
    # 'thorax_bottom':[170,320],
    # 'thorax_midpoint':[0,0],
    # 'left_shoulder_pivot':[385,392],
    # 'right_shoulder_pivot':[25,392]
    # }

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

    # ##necklace11.png
    # jewellery_position={
    # 'thorax_top':[181,207],
    # 'thorax_bottom':[175,384],
    # 'thorax_midpoint':[0,0],
    # 'left_shoulder_pivot':[385,392],
    # 'right_shoulder_pivot':[25,392]
    # }
    
    # ##necklace11.png
    # jewellery_position={
    # 'thorax_top':[175,187],
    # 'thorax_bottom':[175,364],
    # 'thorax_midpoint':[0,0],
    # 'left_shoulder_pivot':[385,392],
    # 'right_shoulder_pivot':[25,392]
    # }



def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    # old (python 2 version):
    # nparr = np.fromstring(encoded_data.decode('base64'), np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    return img





def get_masked_image(jewellery_image,jewellery_position, human_image,RUN_CV_SELFIE_SEGMENTER=True,debug=True):
    imgOut=overlay.get_final_image(jewellery_image,jewellery_position, human_image,RUN_CV_SELFIE_SEGMENTER,debug)
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
    if (request.method=="OPTIONS"):
        return
        
    if (request.method=="POST"):
        content = request.json
        # print("JSON input",file=sys.stderr, flush=True)
        # print(content,file=sys.stderr, flush=True)
        print("Points",file=sys.stderr, flush=True)
        print(content['points'],file=sys.stderr, flush=True)
        jewellery_image= data_uri_to_cv2_img(content["jewellery_image"])
        human_image=data_uri_to_cv2_img(content["human_image"])
        jewellery_position={
            'thorax_top':[float(content['points']['thorax_top_x']),float(content['points']['thorax_top_y'])],
            'thorax_bottom':[float(content['points']['thorax_bottom_x']),float(content['points']['thorax_bottom_y'])],
            }
        print("we have a human image",file=sys.stderr, flush=True) 
        
        
    else:       
        #copy paste values from list of jewellery_position values given above for relvant image
        ## For exmaple, if using necklace11, use thorax_top and thorax_bottom from necklace7 above
        #necklace11
        jewellery_position={
            'thorax_top':[162,180],
            'thorax_bottom':[162,357],
    }
            
        jewellery_image=cv2.imread("./overlay/necklace11.png",cv2.IMREAD_UNCHANGED)
        human_image=cv2.imread("./overlay/human_image19.png",cv2.IMREAD_UNCHANGED)
        
    if (human_image.shape[2]==4):
        print("Converting PNG to BGR")
        human_image=cv2.cvtColor(human_image, cv2.COLOR_BGRA2BGR)
        
    if (human_image.shape[0]>400 and human_image.shape[1]>400):
        human_image=resizeAndPad(human_image,(400,400))
        print("resizing human_image as both >400",file=sys.stderr, flush=True)
    elif (human_image.shape[0]>800 or human_image.shape[1]>800):
        human_image=resizeAndPad(human_image,(500,500))
        print("resizing human_image as one width or heigh>800",file=sys.stderr, flush=True)
        
    # imgOut=overlay.get_sample_preview_image(jewellery_image,jewellery_position,RUN_CV_SELFIE_SEGMENTER=True)
    print(time.time(),file=sys.stderr, flush=True)
    print ("Jewellery Image dimensions:",file=sys.stderr, flush=True)
    print(jewellery_image.shape,file=sys.stderr, flush=True)
    print ("Human Image dimensions:",file=sys.stderr, flush=True)
    print(human_image.shape,file=sys.stderr, flush=True)
    imgOut=get_masked_image(jewellery_image,jewellery_position,human_image,RUN_CV_SELFIE_SEGMENTER=True,debug=False)
    print(time.time(),file=sys.stderr, flush=True)
    # image_url = request.args.get('imageurl')
    # requested_url = urllib.urlopen(image_url)
    # image_array = np.asarray(bytearray(requested_url.read()), dtype=np.uint8)
    # img = cv2.imdecode(image_array, -1)
    # Do some processing, get output_img

    retval, buffer = cv2.imencode('.png', imgOut)
    if (request.method=="POST"):
        buffer_b64encoded = base64.b64encode(buffer)
        data = { 
                "status" : "success", 
                "image" : buffer_b64encoded.decode('utf-8'), 
            }
        return jsonify(data)
    else:
        response=make_response(buffer.tobytes())
        response.headers['Content-Type'] = 'image/png'
        buffer_bytes=buffer.tobytes()
        return response
    
    




@app.route('/preview', methods=['GET','POST','OPTIONS'])
@cross_origin()
def preview():
    
    # print("Headers",file=sys.stderr, flush=True)
    # print(request.headers,file=sys.stderr, flush=True)
    # print("Cookies",file=sys.stderr, flush=True)
    # print(request.cookies,file=sys.stderr, flush=True)
    # print("Data",file=sys.stderr, flush=True)
    # print(request.data,file=sys.stderr, flush=True)
    # print("Args",file=sys.stderr, flush=True)
    # print(request.args,file=sys.stderr, flush=True)
    # print("Form",file=sys.stderr, flush=True)
    # print(request.form,file=sys.stderr, flush=True)
    # print("Endpoint",file=sys.stderr, flush=True)
    # print(request.endpoint,file=sys.stderr, flush=True)
    # print("Method",file=sys.stderr, flush=True)
    # print(request.method,file=sys.stderr, flush=True)
    # print("RemoteAddr",file=sys.stderr, flush=True)
    # print(request.remote_addr,file=sys.stderr, flush=True)
    
    if (request.method=="OPTIONS"):
        return
    
    if (request.method=="POST"):
        content = request.json
        # print("JSON input",file=sys.stderr, flush=True)
        # print(content,file=sys.stderr, flush=True)
        print("Points",file=sys.stderr, flush=True)
        print(content['points'],file=sys.stderr, flush=True)
        jewellery_image= data_uri_to_cv2_img(content["jewellery_image"])
        jewellery_position={
            'thorax_top':[float(content['points']['thorax_top_x']),float(content['points']['thorax_top_y'])],
            'thorax_bottom':[float(content['points']['thorax_bottom_x']),float(content['points']['thorax_bottom_y'])],
            }
    else:
        #necklace3.png
        jewellery_position={
                'thorax_top':[210,307],
                'thorax_bottom':[210,481],
            }
        jewellery_image=cv2.imread("./overlay/necklace3.jpg",cv2.IMREAD_UNCHANGED)
    
    human_image=cv2.imread('./overlay/public3.jpg',cv2.IMREAD_UNCHANGED)
    if (human_image.shape[0]>400 and human_image.shape[1]>400):
        human_image=resizeAndPad(human_image,(400,400))
        print("resizing human_image as both >400",file=sys.stderr, flush=True)
    elif (human_image.shape[0]>600 or human_image.shape[1]>600):
        human_image=resizeAndPad(human_image,(500,500))
        print("resizing human_image as one >600",file=sys.stderr, flush=True)
        
    print(time.time(),file=sys.stderr, flush=True)
    print ("Jewellery Image dimensions:",file=sys.stderr, flush=True)
    print(jewellery_image.shape,file=sys.stderr, flush=True)
    print ("Human Image dimensions:",file=sys.stderr, flush=True)
    print(human_image.shape,file=sys.stderr, flush=True)
    imgOut=overlay.get_sample_preview_image(jewellery_image,jewellery_position,human_image,RUN_CV_SELFIE_SEGMENTER=True)
    print(time.time(),file=sys.stderr, flush=True)
    # imgOut=get_masked_image(jewellery_image,jewellery_position,human_image,RUN_CV_SELFIE_SEGMENTER=True,debug=False)
    # image_url = request.args.get('imageurl')
    # requested_url = urllib.urlopen(image_url)
    # image_array = np.asarray(bytearray(requested_url.read()), dtype=np.uint8)
    # img = cv2.imdecode(image_array, -1)
    # Do some processing, get output_img

    retval, buffer = cv2.imencode('.png', imgOut)
    if (request.method=="POST"):
        buffer_b64encoded = base64.b64encode(buffer)
        data = { 
                "status" : "success", 
                "image" : buffer_b64encoded.decode('utf-8'), 
            }
        return jsonify(data)
    else:
        response=make_response(buffer.tobytes())
        response.headers['Content-Type'] = 'image/png'
        buffer_bytes=buffer.tobytes()
        return response


if __name__ == '__main__':
    # app.run(host='0.0.0.0',port=80,debug=True)
    serve(app, host="0.0.0.0", port=80,threads=10)







