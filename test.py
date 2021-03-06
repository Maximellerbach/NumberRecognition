import keras
from keras.models import load_model
import cv2
from tqdm import tqdm
import numpy as np
import h5py
from glob import glob
import time

#LOAD AI
model = load_model('autre IA\\AI.h5')

#CREATING A LIST CONTAINING INPUT IMAGE'S PATH
dos = glob('image\\*.jpg')

#INITIALIZE THE SIZE OF THE CROPPED IMAGE
wi = 28
he = 28

#SETTINGS FOR THE PUTTEXT FONCTION 
font = cv2.FONT_HERSHEY_SIMPLEX
fontColor = (255,0,0)
lineType = 2


for img_path in tqdm(dos):

    img = cv2.imread(img_path,1)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    th = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,31,24)

    im2, cnts, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cont in tqdm(cnts):
        try:
            x, y, w, h = cv2.boundingRect(cont)
            if h*w>20:
                crop= th[y-4:y+h+4, x-4:x+w+4]
                crop = cv2.resize(img2,(wi,he))
                
                img_pred = np.reshape(crop,(1,wi,he,1))
                
                predicted = np.argmax(model.predict(img_pred))
                cv2.imwrite('C:\\Users\\maxim\\Desktop\\autre IA\\cropped_image\\'+str(predicted)+'_'+str(time.time())+'.png',img2)

                if predicted == 11:
                    fontScale = 1
                    cv2.putText(img, 'M', (x,y+h), font, fontScale, fontColor, lineType)

                if predicted != 10 and predicted!= 11:
                    fontScale = 1
                    cv2.putText(img, str(predicted), (x,y+h), font, fontScale, fontColor, lineType)

                
                
        except:
            pass
            
    cv2.imwrite('image\\'+(img_path.split('\\')[-1]).split('.')[0]+'_'+'.png',img)
    cv2.imwrite('image\\'+(img_path.split('\\')[-1]).split('.')[0]+'_th'+'.png',th)

    
