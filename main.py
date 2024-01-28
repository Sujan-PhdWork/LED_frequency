import cv2
import numpy as np
from crop import MouseEvent #,ref_points
from transformation import Affine_trans
import matplotlib.pyplot as plt


from imutils import contours
from skimage import measure
import numpy as np
import imutils

# plt.ion() 
# graph,=plt.plot([0],[0])

# ref_points

start_playing=False
frames=[]



def process_img(img):
    global graph,t_list,blue_list,t
    # h,w=img.shape
    h,w,_=img.shape
    frames.append(img)
    
    if len(frames)==1:       
       return None

    T=Affine_trans(frames[-2],frames[-1])
    # print(T)
    inv_T=np.linalg.inv(T)
    frames[-1] = cv2.warpAffine(frames[-1], inv_T[:2, :3], (w, h))
    
    image=frames[-1].copy()
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)


    thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY)[1]
    
    # video 3=190,255,1,4
    # video 1=240,255,1,6

    thresh = cv2.erode(thresh, None, iterations=1)
    thresh = cv2.dilate(thresh, None, iterations=6)

    # # to filter out the noise
    labels = measure.label(thresh, connectivity=1, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")

    # # print(labels)
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and count the
        # number of pixels 
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if numPixels > 180: # this can be variable and can be calculated using thresh
            mask = cv2.add(mask, labelMask)
        # cv2.imshow("new",labelMask)
        # print(numPixels)
        # cv2.waitKey(1000)
    # cv2.imshow("masked led",mask)

    # cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
	# cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)
    # cnts = contours.sort_contours(cnts)[0]

    # for (i, c) in enumerate(cnts):
    #     # draw the bright spot on the image
    #     (x, y, w, h) = cv2.boundingRect(c)
    #     ((cX, cY), radius) = cv2.minEnclosingCircle(c)
    #     cv2.circle(image, (int(cX), int(cY)), int(radius),
    #         (0, 0, 255), 3)
    #     cv2.putText(image, "#{}".format(i + 1), (x, y - 5),
    #         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)




    # graph.remove()


    
    cv2.imshow("Improved_image",thresh)
    cv2.waitKey(100)
    
    return frames[-1]







if __name__ =="__main__":
    # Open a video file
    video_capture = cv2.VideoCapture('data/video2.mp4')

    # Check if the video capture object is successfully opened
    if not video_capture.isOpened():
        print("Error: Unable to open video source.")
        exit()

    # Get video properties
    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
    fps = video_capture.get(5)

    # Create VideoWriter object to save the cropped video
    # You can change the codec based on your preference
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    
    
    # stable_video = cv2.VideoWriter(
    #     'result/out_stable.avi', fourcc, fps, (frame_width, frame_height))


    _, frame = video_capture.read()
    Window_Name="Video"
    # cv2.namedWindow(Window_Name)
    event=MouseEvent(frame,Window_Name)
    event.get_coordinates()

    while True:
        ret, frame = video_capture.read()


        if not ret:
            print("Video ended.")
            break
        
        # frame=frame[:,:,0]
        clone = frame.copy()

        # if not start_playing:
            
        
        if not start_playing:
            cv2.imshow(Window_Name, frame)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("p") and len(event.ref_points) == 2:
                start_playing = True
                print("Press 'q' to stop the video.")

        
        # Press 'p' to start playing the video after setting the bounding box
        

        # Break the loop when 'q' key is pressed
        if key == ord("q"):
            break

        # If the bounding box is set, crop and display the video
        if len(event.ref_points) == 2 and start_playing:
            roi = clone[event.ref_points[0][1]:event.ref_points[1]
                        [1], event.ref_points[0][0]:event.ref_points[1][0]]
            
            simg=process_img(roi)
            # h,w=roi.shape
            h,w,_=roi.shape
            if simg is None:
                output_video = cv2.VideoWriter(
                    'result/out.avi', fourcc, fps, (w,h))

                output_video.write(roi)    

                stable_video = cv2.VideoWriter(
                    'result/out_stable.avi', fourcc, fps, (w,h))
                
                stable_video.write(simg)
                continue
            # cv2.imshow("Cropped Video", roi)
            output_video.write(roi)

            stable_video.write(simg)
        
        cv2.waitKey(int(1/fps*1000.0))

    # Release the video capture object, close the windows, and release the VideoWriter
    video_capture.release()
    output_video.release()
    cv2.destroyAllWindows()
    # plt.plot(t_list,blue_list)
    # plt.show()
