import cv2
import numpy as np
from crop import MouseEvent #,ref_points
from transformation import Affine_trans

# ref_points

start_playing=False
frames=[]

params = cv2.SimpleBlobDetector_Params()
# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200
 
# # Filter by Area.
params.filterByArea = True
params.minArea = 100
 
# # Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.8
 
# # Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.95
 
# # # Filter by Inertia
# params.filterByInertia = True
# params.minInertiaRatio = 0.8


def keypoint_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray=img.copy() 
 
    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create(params)
 
    # Detect blobs.
    keypoints = detector.detect(gray)
    return keypoints


def process_img(img):
    # h,w=img.shape
    h,w,_=img.shape
    frames.append(img)
    
    if len(frames)==1:
       return None

    T=Affine_trans(frames[-2],frames[-1])
    # print(T)
    inv_T=np.linalg.inv(T)
    frames[-1] = cv2.warpAffine(frames[-1], inv_T[:2, :3], (w, h))

    
    keypoints=keypoint_detection(frames[-1])
    print(keypoints)

    modified_frame = cv2.drawKeypoints(frames[-1], keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Improved_image",modified_frame)
    # cv2.imshow("Improved_image",frames[-1])

    
    return frames[-1]







if __name__ =="__main__":
    # Open a video file
    video_capture = cv2.VideoCapture('data/video1.mp4')

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
