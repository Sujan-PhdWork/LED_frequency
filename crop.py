import cv2


# start_playing = False
# ref_points=[]

class MouseEvent(object):

    def __init__(self,frame,Window):
        self.start_playing = False
        self.ref_points=[]
        self.frame=frame
        self.Window=Window


    # Global variables
    def click_and_crop(self,event, x, y, flags, param): 
        if not self.start_playing:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.ref_points = [(x, y)]
                cropping = True

            elif event == cv2.EVENT_LBUTTONUP:
                self.ref_points.append((x, y))
                cropping = False

                # Draw a rectangle around the region of interest
                cv2.rectangle(self.frame, self.ref_points[0], self.ref_points[1], (0, 255, 0), 2)
                cv2.imshow(self.Window, self.frame)
    
    def get_coordinates(self):
        cv2.namedWindow(self.Window)
        cv2.setMouseCallback(self.Window, self.click_and_crop)


