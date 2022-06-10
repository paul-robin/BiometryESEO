import cv2
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from random import random
from image_recog import auth_main

def process_mains(img, finalPath):
    authPath = os.getcwd() + "\\auth_mains"

    gray = cv2.GaussianBlur(img, (3,3), 0)
    ridge_filter = cv2.ximgproc.RidgeDetectionFilter_create()
    ridges = ridge_filter.getRidgeFilteredImage(gray)
    ridges = cv2.bitwise_not(ridges)

    ridges = cv2.resize(ridges, (128, 128))

    cv2.imwrite(finalPath + "\\" + str(i) + "_0.jpg", ridges)


def create_dataset(img):
    finalPath = os.getcwd() + "\\dataset_mains"
    for j in range(1, 20):
        pts1 = np.float32([[random()*12,       random()*12],
                            [116+(random()*12), random()*12], 
                            [random()*12,       116+(random()*12)], 
                            [116+(random()*12), 116+(random()*12)]])

        pts2 = np.float32([[0,   0],
                            [128, 0],
                            [0,   128],
                            [128, 128]])
        
        M = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(img,M,(128,128))

        lookUpTable = np.empty((1,256), np.uint8)
        gamma = 0.75+random()*0.5
        for k in range(256):
            lookUpTable[0,k] = np.clip(pow(k / 255.0, gamma) * 255.0, 0, 255)
        res = cv2.LUT(dst, lookUpTable)

        gray = cv2.GaussianBlur(res, (3,3), 0)
        ridge_filter = cv2.ximgproc.RidgeDetectionFilter_create()
        ridges = ridge_filter.getRidgeFilteredImage(gray)
        ridges = cv2.bitwise_not(ridges)

        cv2.imwrite(finalPath + "\\" + str(i) + "_" + str(j) + ".jpg", ridges)


#----------------------------------------------------------------#


basePath = os.getcwd() + "\\bases_mains"
todoPath = os.getcwd() + "\\auth_todo"
files = [f for f in listdir(basePath) if isfile(join(basePath, f))]
filesToAuth = [f for f in listdir(todoPath) if isfile(join(todoPath, f))]


# Dataset setup
for i in range(len(files)):
    process_mains(cv2.imread(basePath + '\\' + files[i]), os.getcwd() + "\\dataset_mains")
    create_dataset(cv2.imread(basePath + '\\' + files[i]))


# Auth process + test
for i in range(len(filesToAuth)):
    process_mains(filesToAuth[i], os.getcwd() + "\\auth_tmp")

for file in filesToAuth:
    auth_main(todoPath + "//" + file)