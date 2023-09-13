
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2



def detect_show_image(image_path):
    """
    this function takes an image_path and return detected objects on the image
    """
    img=cv2.imread(image_path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = detect(img)
    return plt.imshow(np.squeeze(results.render()))


def cropped_images(detect ,original_img_path):
    """
    this function takes an image_path 
    then returns each detected object as a cropped version of the original image
    """
    images_list=[]
    img=cv2.imread(original_img_path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = detect(img)
    for i in range(len(results.crop(save=False))):
        ymin=int(results.pandas().xyxy[0]['ymin'][i])
        ymax=int(results.pandas().xyxy[0]['ymax'][i])
        xmin=int(results.pandas().xyxy[0]['xmin'][i])
        xmax=int(results.pandas().xyxy[0]['xmax'][i])
        images_list.append(img[ymin:ymax,xmin:xmax])
    return images_list   
def numberOfDetections(im_path):
    img=cv2.imread(im_path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = detect(img)
    return(len(results.crop(save=False)))






if __name__ == '__main__':
    checkPoint_path=r'C:\Users\Farhang\Desktop\visual-search\Yolo_weights\exp2\weights\best.pt'
    detect = torch.hub.load('ultralytics/yolov5', 'custom', path=checkPoint_path)
    path = r"C:\Users\Farhang\Desktop\visual-search\model\WIN_20230324_18_28_34_Pro.jpg"
    
    detect_show_image(path)

