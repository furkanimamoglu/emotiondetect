import argparse as argparse
import cv2
# import vlc
import random
import time

import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from pyimagesearch import LeNet
import os

random.seed(61)

# def PlaySong(song):
#     vlc_instance = vlc.Instance()
#     song = song
#     player = vlc_instance.media_player_new()
#     media = vlc_instance.media_new(song)
#
#     media.get_mrl()
#     player.set_media(media)
#
#     player.play()
#     playing = set([1])
#     time.sleep(1.5)  # startup time.
#     duration = player.get_length() / 1000
#     mm, ss = divmod(duration, 60)
#
#     print ("Playing song right now: ", song, "Duration:", "%02d:%02d" % (mm, ss))
#
#     time_left = True
#
#     while time_left == True:
#
#         song_time = player.get_state() #VLC player'in güncel state durumu kontrol edilir
#
#         print ('State: %s' % song_time)
#
#         if song_time not in playing:
#             time_left = False
#
#         time.sleep(duration)  #Şarkı süresi boyunca program bekletilir.
#
#     print ('I hope you enjoyed it?')


def SelectMood(songtype):
    if(songtype=="Happy"):
        randomnumber = random.randint(1,5)
        song = f"playlist/happy/{randomnumber}.mp3"
        print("Happy")
    elif(songtype=="Sad"):
        print("Sad")
        randomnumber = random.randint(1, 5)
        song = f"playlist/sad/{randomnumber}.mp3"
        print("Sad")
    else:
        print("Error")

#Ayar Değişkenler
playingsong=0
songstartedtoplay=0

#Haar Dataset: insan yüzleri, gülümsemeleri, gözleri içeren datasetleri
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_smile.xml')

#Webcam
webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()

    if not successful_frame_read:
        print("Hata: Frame okunamadi")
        break


    # #GrayScale
    # frame_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #
    # #İnsan yüzünün ve gülümsemesinin tespit edilmesi
    # faces = face_detector.detectMultiScale(frame_grayscale)
    # if len(faces) > 0:
    #     print(faces[-1])
    #     #print(smiles[-1])
    # #Drawing Square
    # for(x,y,w,h) in faces:
    #     cv2.rectangle(frame, (x,y), (x+w, y+h), (100,200,50),4)
    #     the_face = frame[y:y+h, x:x+w]
    #
    #     # GrayScale
    #     face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
    #
    #     smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=20)
    #
    #     for(x_,y_,w_,h_) in smiles:
    #
    #         cv2.rectangle(the_face, (x_, y_), (x_ + w_, y_ + h_), (50, 50, 200), 4)
    #
    #
    #
    # if len(smiles) > 0:
    #     cv2.putText(frame,'Gulumseme',(x, y+h+20), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,255,255))
    #     SelectMood("Happy")
    #
    # cv2.imshow('Smily',frame)
    #
    # if cv2.waitKey(1) & 0xFF == ord('s'):
    #     print("Next song is coming!.")
    #
    # if cv2.waitKey(1) & 0xFF == ord('f'):
    #     print("See you later.")
    #     break

#Cleanup
webcam.release()
cv2.destroyAllWindows()