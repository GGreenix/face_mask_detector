import cv2
import numpy as np 

def get_avg_color_from_forehead(forehead):
    avg_h,avg_s,avg_v = 0,0,0
    reshaped_forehead = forehead.shape[0]*forehead.shape[1]
    flatened_forehead = np.reshape(forehead,(reshaped_forehead,3))
    #print (flatened_forehead.shape)
    for pixel in flatened_forehead:
        avg_h += pixel[0]
        avg_s += pixel[1]
        avg_v += pixel[2]
    
    avg_h = int(avg_h/reshaped_forehead)
    avg_s = int(avg_s/reshaped_forehead)
    avg_v = int(avg_v/reshaped_forehead)

    return (avg_h,avg_s,avg_v)

def main():
    cap = cv2.VideoCapture(0)
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    while(cap.isOpened()):
        _, frame = cap.read()
        gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_frame, 1.3, 5)
        for x,y,w,h in faces:
            forehead = frame[y+int(h*0.09):y+int(h*0.24),x+int(w*0.1):x+int(w*0.8)]
            avg_h,avg_s,avg_v = get_avg_color_from_forehead(forehead)
            range_factor = 35
            low_skin_color = (avg_h-range_factor,avg_s-range_factor,avg_v-range_factor)
            high_skin_color = (avg_h+range_factor,avg_s+range_factor,avg_v+range_factor)
            nose_to_chin = frame[int(y + h*0.5):y+h,x:x+w]
            
            kernel = np.ones((5,5),'uint8')
            threshsv = cv2.inRange(nose_to_chin,low_skin_color,high_skin_color)
            threshsv = cv2.erode(threshsv,kernel,iterations=1)
            
            thresh_flat =threshsv.flatten()
            amount_of_white = sum(thresh_flat == 255)
            
            ratio_between_pixels_to_white = threshsv.size/amount_of_white
            print(ratio_between_pixels_to_white)
            if ratio_between_pixels_to_white > 15:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            else:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.imshow("dd",threshsv)
            #cv2.imshow("lol",forehead)
            

            
            
        cv2.imshow('Frame',frame)
                
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()