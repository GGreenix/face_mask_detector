import cv2
import numpy as np

class Face:
    def __init__(self,x,y,w,h,frame):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.image = frame

    def get_avg_color_from_forehead(self):
        forehead = self.get_forehead()
        avg_b,avg_g,avg_r = 0,0,0
        
        reshaped_forehead = forehead.shape[0]*forehead.shape[1]
        flatened_forehead = np.reshape(forehead,(reshaped_forehead,3))
        
        for pixel in flatened_forehead:
            avg_b += pixel[0]
            avg_g += pixel[1]
            avg_r += pixel[2]
        
        avg_b = int(avg_b/reshaped_forehead)
        avg_g = int(avg_g/reshaped_forehead)
        avg_r = int(avg_r/reshaped_forehead)

        return (avg_b,avg_g,avg_r)

    def get_mouth_area(self):
        return self.image[int(self.y + self.h*0.5):self.y+self.h,self.x:self.x+self.w]

    def get_forehead(self):
        return self.image[self.y+int(self.h*0.09):self.y+int(self.h*0.24),self.x+int(self.w*0.1):self.x+int(self.w*0.8)]
        
        
    