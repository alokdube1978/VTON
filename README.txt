===Installation Instructions (API based usage explained below)===

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
   	jewellery_image=base64encode(image) //this is the base64 encoded value of the jewellery image including the 							//preceding "data:image/png;base64,.."
    human_image=base64encode(human_image) //this is the base64 encoded value of the human image including the preceding 						//"data:image/png;base64,.." , can be captured from webcam..
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


