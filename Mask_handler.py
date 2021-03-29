from Face import Face
import numpy as np
import cv2
import os

class Mask_handler:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.face_classifier = cv2.CascadeClassifier(os.path.join('cascades','haarcascade_frontalface_default.xml'))
        self.range_factor = 35
        self.kernel = np.ones((5,5),'uint8')
        self.min_ratio_between_white_pixels_to_image_size = 15


   

    def start_cam(self):
        while(True):
            _, frame = self.cap.read()

            gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = self.face_classifier.detectMultiScale(gray_frame, 1.3, 5)
            for x,y,w,h in faces:
                new_face = Face(x,y,w,h,frame)
                avg_b,avg_g,avg_r = new_face.get_avg_color_from_forehead()
            
                low_skin_color = (avg_b-self.range_factor,avg_g-self.range_factor,avg_r-self.range_factor)
                high_skin_color = (avg_b+self.range_factor,avg_g+self.range_factor,avg_r+self.range_factor)
            
            
                nose_to_chin = new_face.get_mouth_area()
                threshsv = cv2.inRange(nose_to_chin,low_skin_color,high_skin_color)
                threshsv = cv2.erode(threshsv,self.kernel,iterations=1)
                
                thresh_flat =threshsv.flatten()
                amount_of_white = sum(thresh_flat == 255)
                
                ratio_between_pixels_to_white = threshsv.size/amount_of_white
                
                if ratio_between_pixels_to_white > self.min_ratio_between_white_pixels_to_image_size:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)    #green ractangle if mask is on properly
                else:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)    #red rectangle if mask is not on properly
                cv2.imshow("thresholded image",threshsv)
                
                

                
                
                cv2.imshow('Frame',frame)
                    
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break
            
        cap.release()
        cv2.destroyAllWindows()