====Mechanism - see VTON-Final PPTx to get an idea of the mechanism used to superimpose necklaces===

===Installation Instructions (API based usage explained furtner below)===

This SECTION is for installing the server on a new node - and not for API based usage

Run pip install -r requirements.txt to get all dependencies

all paths here are Windows based paths - please change them to Linux based

Refer
https://developers.google.com/mediapipe/solutions
https://docs.opencv.org/4.x/d9/df8/tutorial_root.html
also note all image arrays are of the form [y][x] and not [x][y] for pixel points 


How to Run:
sudo python function_wrappers.py

Runs flask on port 80, please ensure it runs with sudo or change port in function_wrappers.py


===================================================

 
===API based usage===

API based usage:

Note that all APIs are POST based - 

the GET methods on the APIs only return output on stock sample images - as placeholder

The APIs are hosted on vton.embitel.xyz

The "Preview Image" - to check if Jewellery thorax top and bottom are properly defined is at:

http://vton.embitel.xyz/preview

Sample Javascript Code params for usage:
    points['thorax_top_x'] = X_thorax_top;
    points['thorax_top_y'] = Y_thorax_top;
    points['thorax_bottom_x'] = X_thorax_bottom;
    points['thorax_bottom_y'] = Y_thorax_bottom;
   	jewellery_image=base64encode(image) //this is the base64 encoded value of the jewellery image including the 							//preceding "data:image/png;base64,.."
    	$.ajax({
            url: "http://vton.embitel.xyz/preview",
            type: "POST",
			         contentType: "application/json", //this is mandatory
			         dataType: "json",
		         	data: JSON.stringify({                     
                        points: points,
                        jewellery_image: jewellery_image                        

                    }),
        }).done(function( data, textStatus, jqXHR ) {
			console.log(data);
            $("#image").attr('src', 'data:png/jpeg;base64,' + data.image); ///send the preview image back to front preview 									//image placeholder
        }).fail(function( jqXHR, textStatus, errorThrown ) {
            alert("fail: " + errorThrown);
        });

====Actual overlay on human image====
The "final Image" - to overlay Jewellery on Human Image is defined at:
http://vton.embitel.xyz/overlayimage
Sample code for actual overlay:
    points['thorax_top_x'] = X_thorax_top;
    points['thorax_top_y'] = Y_thorax_top;
    points['thorax_bottom_x'] = X_thorax_bottom;
    points['thorax_bottom_y'] = Y_thorax_bottom;
   	jewellery_image=base64encode(image) //this is the base64 encoded value of the jewellery image including the 							
						//preceding "data:image/png;base64,.."
    human_image=base64encode(human_image) //this is the base64 encoded value of the human image including the preceding 						
						//"data:image/png;base64,.." , can be captured from webcam..
    $.ajax({
            url: "http://vton.embitel.xyz/overlayimage",
            type: "POST",
			         contentType: "application/json", //this is mandatory
			         dataType: "json",
		         	data: JSON.stringify({                     
                        points: points,
                        jewellery_image: jewellery_image,
                        human_image: human_image

                    }),
        }).done(function( data, textStatus, jqXHR ) {
			console.log(data);
            $("#image").attr('src', 'data:png/jpeg;base64,' + data.image); ///send the final image back to front preview 										//image placeholder
        }).fail(function( jqXHR, textStatus, errorThrown ) {
            alert("fail: " + errorThrown);
        });


===================================================================================


The server is a flask wrapped server started via:
nohup sudo python function_wrappers.py &

The server listens on port 80 

and processes the incoming data and overlays the jewellery_image on a demo image/human_image 


/preview
for preview image - when marking thorax points via admin panel

/overlayimage
it overlays jewellery_image on the final provided human_image when using the overlayimage end point


Response Format:

status field in the response indicates "success" or "error"


A working response will be:

{ 
"image":"(base64 overlayed image)",
"message": "",
 "status": "success"
} 

if there is an error , "image" will not be returned, instead we will have 
{ 
"message": "(error message)",
 "status": "error"
}

Possible error messages are::

"Unable to extract data from input" -- occurs when the params are wrong/not set/not parseable

"Invalid Coordinates for Jewellery" -- if jewellery thorax coordinates are wrong like 0,0 etc or outside jewellery pic dimensions

"Unable to determine Jewellery points" -- if not able to parse the jewellery image due to bad image

"Unable to extract Facial Landmark points from human image" --- if we cannot determine thorax points or shoulder points from given human image
