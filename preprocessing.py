import numpy as np
import glob
import cv2 as cv
import random
from tqdm import tqdm 

class ImageViewer:
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.jpg_files = glob.glob(img_dir + '/*.jpg')
        self.img_path = None
        self.adaptive_thresh = None
        self.binary_thresh = None

    def select_random_image(self):
        import random
        random_num = random.randint(0, len(self.jpg_files)-1)
        self.img_path = self.jpg_files[random_num]

    def show_image(self):
        if self.img_path is None:
            print("No image selected")
            return
        img = cv.imread(self.img_path)
        cv.imshow('Image', img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def len_images(self):
        return len(self.jpg_files)
    
    def list_images(self):
        return self.jpg_files
    
    def binary_threshold(self):
        if self.img_path is None:
            print("No image selected")
            return
        img = cv.imread(self.img_path, 0)
        ret, self.binary_thresh = cv.threshold(img, 253, 255, cv.THRESH_BINARY)

    def adaptive_threshold(self):
        if self.img_path is None:
            print("No image selected")
            return
        img = cv.imread(self.img_path, 0)
        self.adaptive_thresh = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)

    def show_image_with_threshold(self):
        if self.img_path is None:
            print("No image selected")
            return
        img = cv.imread(self.img_path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        self.binary_thresh = cv.threshold(gray, 253, 255, cv.THRESH_BINARY )[1]
        self.adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
        img_combined = np.concatenate((gray, self.binary_thresh, self.adaptive_thresh), axis=1)
        resized_img = cv.resize(img_combined, (1000, 500))
        cv.imshow('Image / Binary / Adaptive Thresholded', resized_img)
        cv.waitKey(0)
        cv.destroyAllWindows()

# imgviewer = ImageViewer(img_dir)
# imgviewer.select_random_image()
# imgviewer.show_image_with_threshold()









class Preprocessing:
    def __init__(self, img_dir, output_dir , background_dir=None):

        self.jpg_files = glob.glob(img_dir + '/*.jpg')
        self.output_dir = output_dir
        self.background_dir = glob.glob(f'{background_dir}/*.jpg') if background_dir else None 

    def process_image(self , img_path):
        img = cv.imread(img_path)
        blur_img = cv.GaussianBlur(img, (11, 11), 0)
        img_gray = cv.cvtColor(blur_img , cv.COLOR_BGR2GRAY)
         
        img_threshold = cv.threshold(img_gray, 254, 255, cv.THRESH_BINARY)[1]
        img_threshold_not = cv.bitwise_not(img_threshold)
        kernel = np.ones((5, 5), np.uint8)
        img_closing = cv.morphologyEx(img_threshold_not, cv.MORPH_CLOSE, kernel)

        img_mask = cv.bitwise_and(img ,img,  mask=img_closing)
        


        return img_mask

    
    def show_result(self):
        import random
        img = self.process_image(self.jpg_files[random.randint(0, len(self.jpg_files)-1)])
        path = self.jpg_files[random.randint(0, len(self.jpg_files)-1)]
        print(path.split()[-1][:-4])
        cv.imshow("img with pp" , img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    def save_pp(self):
        for path in self.jpg_files:

            img_path = self.output_dir + r"\image_pp" +str(path.split()[-1][:-4]) + ".jpg" 
            img = self.process_image(path)
            cv.imwrite(img_path , img)
    
    def process_image_trans(self, img_path):
        img = cv.imread(img_path)

        rgba = cv.cvtColor(img , cv.COLOR_BGR2RGBA)

        blur_img = cv.GaussianBlur(img, (11, 11), 0)
        img_gray = cv.cvtColor(blur_img , cv.COLOR_BGR2GRAY)
         
        img_threshold = cv.threshold(img_gray, 254, 255, cv.THRESH_BINARY)[1]
        img_threshold_not = cv.bitwise_not(img_threshold)
        kernel = np.ones((5, 5), np.uint8)
        img_closing = cv.morphologyEx(img_threshold_not, cv.MORPH_CLOSE, kernel)
        alpha = img_closing.copy()
        alpha[alpha > 0] = 255
        rgba[:, :, 3] = alpha
        return rgba

        


    def save_pp_trance(self):
        for path in self.jpg_files:
            img_path = self.output_dir + r"\image_pp_tranc_" +str(path.split()[-1][:-4]) + ".jpg" 
            img   = self.process_image_trans(path)
            cv.imwrite(img_path , img)



    def combinde_img(self):
        if self.background_dir == None:
            print("there is no path for background")
            return
        
        for path in tqdm(self.jpg_files):
            random_num = random.randint(0, len(self.background_dir)-1)
            
            foreground = cv.imread(path)
            background = cv.imread(self.background_dir[random_num])
            blur_img = cv.GaussianBlur(foreground, (11, 11), 0)
            img_gray = cv.cvtColor(blur_img , cv.COLOR_BGR2GRAY)
                
            img_threshold = cv.threshold(img_gray, 254, 255, cv.THRESH_BINARY)[1]
            img_threshold_not = cv.bitwise_not(img_threshold)
            kernel = np.ones((5, 5), np.uint8)
            img_closing = cv.morphologyEx(img_threshold_not, cv.MORPH_CLOSE, kernel)
            erosion = cv.erode(img_closing , kernel , iterations = 5)
            alpha = erosion.copy()
            alpha[alpha > 0] = 255
            
            background = cv.resize(background, (foreground.shape[1], foreground.shape[0]))



            foreground = cv.bitwise_and(foreground, foreground, mask=alpha)
            background = cv.bitwise_and(background,  background, mask=cv.bitwise_not(alpha))
            
            result = cv.addWeighted(foreground, 1, background, 1, 0)
            img_path = self.output_dir + r"\image_pp_background_" +str(path.split()[-1][:-4]) + ".jpg" 
            cv.imwrite(img_path, result)


            

        


        
img_dir = r"C:\Users\Farhang\Desktop\visual-search\orginal_imgs"
background_dir = r'C:\Users\Farhang\Desktop\visual-search\background'
out_dir = r'C:\Users\Farhang\Desktop\visual-search\tarkib'

pp = Preprocessing(img_dir ,output_dir=out_dir  , background_dir=background_dir)
print("start")
pp.combinde_img()
print("done")