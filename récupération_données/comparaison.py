# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from PIL import Image
import os 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re

f1 = open("signature.txt", "r")   
f2 = "qmfjlqklqncn"  
  
i = 0
  
for line1 in f1.readlines(): 
    i += 1
      
    if line1.strip() == f2:   
        
        img = mpimg.imread('autorise.jpg')
        imgplot = plt.imshow(img)
        plt.show()        
    else: 
        img = mpimg.imread('refuse.jpg')
        imgplot = plt.imshow(img)
        plt.show()
    break
  
f1.close()                                        
   
