# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


from PIL import Image
import os 
import matplotlib.pyplot as plt

#im.show()

chemin = "dataset"
liste =os.listdir(chemin)
imlist = []
print(list)
for file in  liste:
    fig = plt.figure()

    imlist.append(Image.open(os.path.join(chemin, file)))

    """print(os.path.join(chemin,file))
    handprint_database_image = plt.imshow(im)
    """

print(imlist[0])    
#for i in range(len(list)):
    #im1 = im.read(list[i], im.
    

     
"""
r= 1
t= 2
v = r
if v = 1 :
    autorise = Image.open('autorise.jpg')
    autorise.show()
    else if v = 2 :
        refuse = Image.open('refuse.jpg')
        refuse.show()
 """       
    