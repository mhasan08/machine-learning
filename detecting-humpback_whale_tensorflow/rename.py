import os
import cv2
import time

def renameImage():
    pic_num = 1

    for imageName in os.listdir("./humpback_raw"):
        print imageName
        img = cv2.imread("humpback_raw/"+imageName)
        cv2.imwrite("original_data/"+str(pic_num)+".jpg", img)
        time.sleep(0.25)
        #os.remove("pics2/"+imageName)
        pic_num = pic_num + 1
        
renameImage()
