import cv2

# ref_points = []
# cropping = False
# start_playing = False

def MouseEvent(Window_Name,frame,start_playing):
    # Global variables
    ref_points=[] 
    cropping=False
    
    def click_and_crop(event, x, y, flags, param):
        

        if not start_playing:
            if event == cv2.EVENT_LBUTTONDOWN:
                ref_points = [(x, y)]
                cropping = True

            elif event == cv2.EVENT_LBUTTONUP:
                ref_points.append((x, y))
                cropping = False

                # Draw a rectangle around the region of interest
                cv2.rectangle(frame, ref_points[0], ref_points[1], (0, 255, 0), 2)
                cv2.imshow(Window_Name, frame)
    

    
    
    cv2.setMouseCallback(Window_Name, click_and_crop)
    return ref_points


    