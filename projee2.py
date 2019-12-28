# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas 
import os 
from random import shuffle
import cv2
  

TEST_KLASORU= '/home/busra/Masaüstü/dogs-vs-catss/test1' 
EGITIM_KLASORU = '/home/busra/Masaüstü/dogs-vs-catss/train'
OGRENME_ORANI= 1e-3 #0.001
MODEL_ADI= "dogs-vs-cats-{}-{}.model".format(OGRENME_ORANI,"6conv-fire")
RESIM_BOYUTU = 50

#! /usr/bin/env python

def etiket_olustur(resim):           #goruntu adını bir diziye dönüştürmekte
    resim_adi= resim.split(".")[-3] #dosya adında kullanılan "cat" yada "dog" kelimelerini al
    if resim_adi == "cat": 
        return [1,0]     #fonksiyon dosya adı "cat" bu cıkısı verir.
    elif resim_adi == "dog":
        return [0,1]
        #RESİMLERİN MATRİS HALİNE DONUSMESİ#

def train_data_loder():
    egitim_verisi_2 = []
    for img in tqdm(os.listdir(EGITIM_KLASORU)):
        img_lable = etiket_olustur(img)
        path_to_img = os.path.join(EGITIM_KLASORU,img)
        img = cv2.resize(cv2.imread(path_to_img,cv2.IMREAD_GRAYSCALE),(RESIM_BOYUTU,RESIM_BOYUTU))#resimler gri olarak okunup 50*50 piksel olacak sekilde yeniden boyutlandırılır
        egitim_verisi_2.append([np.array(img),np.array(img_lable)])
        
    shuffle(egitim_verisi_2)#fonksiyon içindeki verilerin karsılastırılması saglanır
    np.save("training_data_new.npy",egitim_verisi_2)# olusturulan egitim verisi egitim_verisi.npy isimli dosyaya yazılır
    return egitim_verisi_2


def testing_data():  #test klasorundeki resimlerdem egitimde kullanılabilecek sekilde test verisi olusturur.
    test_data = []
    for img in tqdm(os.listdir(TEST_KLASORU)):
        img_labels = img.split(".")[0]
        path_to_img = os.path.join(TEST_KLASORU,img)
        img = cv2.resize(cv2.imread(path_to_img,cv2.IMREAD_GRAYSCALE),(RESIM_BOYUTU,RESIM_BOYUTU))
        test_data.append([np.array(img),np.array(img_labels)])
        
    shuffle(test_data)
    np.save("test_dataone.npy",test_data) #olusturulan test verisi olusturulan dosyaya yazılır
    return test_data
train_data_loder()
train_data_g = np.load('training_data_new.npy')



import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression

import tensorflow as tf
 
# MİMARİNİN OLUSURULMASI
tf.reset_default_graph()
#agın giriş boyutlarıının ne olacagı tanımlanır
convnet = input_data(shape=[None, RESIM_BOYUTU, RESIM_BOYUTU, 1], name='input')
#32 adet 5*5 boyutunda filtrelerden olusan ve relu aktivasyonu konvolusyon katmanı
convnet = conv_2d(convnet, 32, 5, activation='relu')
#5*5 boyutunda filtrelereden olusan katman
convnet = max_pool_2d(convnet, 5)
# Layer 2
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
# Layer 3
convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
# Layer 4
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
# Layer 5
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
#1024 birimden olusan tam baglantılı ve relu aktivasyonlu katman
convnet = fully_connected(convnet, 1024, activation='relu')
#ezberlemeyi engellemek için dropout katmanı
convnet = dropout(convnet, 0.8)
#iki birimli ve softmax aktivasyonlu tam baglantılı katman
convnet = fully_connected(convnet, 2, activation='softmax')
#olusturulan mimariyi,ogrenme oranını,optimizasyon,optimizayon turunu,kayıp fonksiyonunu ve dosya isimlerinden aldıgımz hedef  aldıgımız hedef degeri
#kullanarak agı olustur
convnet = regression(convnet, optimizer='adam', learning_rate=OGRENME_ORANI, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

#model.fit (X, Y, n_epoch = 6, validation_set = (test_x, test_y),görüntü_step = 500, show_metric = Doğru , run_id = MODEL_NAME)

if os.path.exists("{}.meta".format(MODEL_ADI)):
    model.load(MODEL_ADI)
    print("Model Loaded")

     #agı eğitirken 12500 adet resmi egitimi test etmek için kullanacagız

egitim = train_data_g[:-12500]
test = train_data_g[-12500:]

X_egitim = np.array([i[0] for i in egitim]).reshape(-1, RESIM_BOYUTU, RESIM_BOYUTU, 1)
y_egitim = [i[1] for i in egitim]
X_test = np.array([i[0] for i in test]).reshape(-1, RESIM_BOYUTU, RESIM_BOYUTU, 1) 
y_test = [i[1] for i in test]

model.fit(X_egitim, y_egitim, n_epoch=6, validation_set=(X_test,  y_test),
    snapshot_step=500, show_metric=True, run_id=MODEL_ADI)

model.save(MODEL_ADI)
test_data = np.load("test_dataone.npy")


figs = plt.figure()
testing_data()
for num,data in enumerate(test_data[:10]):
    test_img = data[0]
    test_lable = data[1]
    test_img_feed = test_img.reshape(RESIM_BOYUTU,RESIM_BOYUTU,1)
    t = figs.add_subplot(3,4,num+1)
    ores = test_img
    model_pred = model.predict([test_img_feed])[0]
    if np.argmax(model_pred) == 1:
        pred_val = "Dog"
    else:
        pred_val = "Cat"
        
    t.imshow(ores,cmap="None")
    plt.title(pred_val)

    t.axes.get_xaxis().set_visible(False)
    t.axes.get_yaxis().set_visible(False)
plt.show()