from tensorflow.python.client import device_lib
import os
import time
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from pylab import *

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.datasets import mnist

from keras.layers import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform,he_uniform

from keras.layers import Layer
from keras.regularizers import l2
from keras import backend as K
from keras.utils import plot_model,normalize

from sklearn.metrics import roc_curve,roc_auc_score

from fv_utils import *
from model import *


#----------------------------------------------------------------#


class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)
    
    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = K.sum(K.square(anchor-positive), axis=-1)
        n_dist = K.sum(K.square(anchor-negative), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)
    
    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss


#----------------------------------------------------------------#
#Neural network :


def DrawPics(tensor,nb=0,template='{}',classnumber=None):
    if (nb==0):
        N = tensor.shape[0]
    else:
        N = min(nb,tensor.shape[0])
    fig=plt.figure(figsize=(16,2))
    nbligne = floor(N/20)+1
    for m in range(N):
        subplot = fig.add_subplot(int(nbligne),min(N,20),m+1)
        axis("off")
        plt.imshow(tensor[m,:,:,0],vmin=0, vmax=1,cmap='Greys')
        if (classnumber!=None):
            subplot.title.set_text((template.format(classnumber)))


def build_model(input_shape, network, margin=0.2):
    '''
    Define the Keras Model for training 
        Input : 
            input_shape : shape of input images
            network : Neural network to train outputing embeddings
            margin : minimal distance between Anchor-Positive and Anchor-Negative for the lossfunction (alpha)
    '''

     # Define the tensors for the three input images
    anchor_input = Input(input_shape, name="anchor_input")
    positive_input = Input(input_shape, name="positive_input")
    negative_input = Input(input_shape, name="negative_input") 
    
    # Generate the encodings (feature vectors) for the three images
    encoded_a = network(anchor_input)
    encoded_p = network(positive_input)
    encoded_n = network(negative_input)
    
    #TripletLoss Layer
    loss_layer = TripletLossLayer(alpha=margin,name='triplet_loss_layer')([encoded_a,encoded_p,encoded_n])
    
    # Connect the inputs with the outputs
    network_train = Model(inputs=[anchor_input,positive_input,negative_input],outputs=loss_layer)
    
    # return the model
    return network_train


#----------------------------------------------------------------#
#Data batch prep :


def get_batch_random(batch_size,s="train"):
    """
    Create batch of APN triplets with a complete random strategy
    
    Arguments:
    batch_size -- integer 

    Returns:
    triplets -- list containing 3 tensors A,P,N of shape (batch_size,w,h,c)
    """
    if s == 'train':
        X = x_train
    else:
        X = x_test

    m, w, h,c = X[0].shape
    
    
    # initialize result
    triplets=[np.zeros((batch_size,h, w,c)) for i in range(3)]
    
    for i in range(batch_size):
        #Pick one random class for anchor
        anchor_class = np.random.randint(0, nb_classes)
        nb_sample_available_for_class_AP = X[anchor_class].shape[0]
        
        #Pick two different random pics for this class => A and P
        [idx_A,idx_P] = np.random.choice(nb_sample_available_for_class_AP,size=2,replace=False)
        
        #Pick another class for N, different from anchor_class
        negative_class = (anchor_class + np.random.randint(1,nb_classes)) % nb_classes
        nb_sample_available_for_class_N = X[negative_class].shape[0]
        
        #Pick a random pic for this negative class => N
        idx_N = np.random.randint(0, nb_sample_available_for_class_N)

        triplets[0][i,:,:,:] = X[anchor_class][idx_A,:,:,:]
        triplets[1][i,:,:,:] = X[anchor_class][idx_P,:,:,:]
        triplets[2][i,:,:,:] = X[negative_class][idx_N,:,:,:]

    return triplets


def drawTriplets(tripletbatch, nbmax=None):
    """display the three images for each triplets in the batch
    """
    labels = ["Anchor", "Positive", "Negative"]

    if (nbmax==None):
        nbrows = tripletbatch[0].shape[0]
    else:
        nbrows = min(nbmax,tripletbatch[0].shape[0])
                 
    for row in range(nbrows):
        fig=plt.figure(figsize=(16,2))
    
        for i in range(3):
            subplot = fig.add_subplot(1,3,i+1)
            axis("off")
            plt.imshow(tripletbatch[i][row,:,:,0],vmin=0, vmax=1,cmap='Greys')
            subplot.title.set_text(labels[i])


#----------------------------------------------------------------#
#Data Validation :


def verify(image_path, identity, database, model):
    """
    Function that verifies if the person on the "image_path" image is "identity".
    
    Arguments:
    image_path -- path to an image
    identity -- string, name of the person you'd like to verify the identity. Has to be a resident of the Happy house.
    database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
    model -- your Inception model instance in Keras
    
    Returns:
    dist -- distance between the image_path and the image of "identity" in the database.
    door_open -- True, if the door should open. False otherwise.
    """
    
    # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
    encoding = img_to_encoding(image_path, model)
    
    # Step 2: Compute distance with identity's image (≈ 1 line)
    dist = np.linalg.norm(encoding-database[identity])
    
    # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
    if dist < 0.7:
        print("It's " + str(identity) + ", it's a match", end=" ")
        door_open = True
    else:
        print("It's not " + str(identity) + ", fingerprint mismatch", end=" ")
        door_open = False
        
    return dist, door_open


#----------------------------------------------------------------#


nb_classes = 12
dataPath = os.getcwd() + "//dataset_mains//"
x_train,y_train,x_test,y_test,testfiles = get_data_label(dataPath,ratio = 0.2)

# print(testfiles)

print(f'Checking shapes for class {y_train[0]} (train) : ',x_train[0].shape)
print(f'Checking shapes for class {y_test[0]} (test) : ',x_test[0].shape)
print("Checking first samples")
for i in range(2):
    DrawPics(x_train[i],5,template='Train {}',classnumber=y_train[i])
    DrawPics(x_test[i],5,template='Test {}',classnumber=y_test[i])

input_shape=(128, 128, 3)
FRmodel = palmRecoModel(input_shape = input_shape, embeddingsize = 128)
network_train = build_model(input_shape,FRmodel)
optimizer = Adam(lr = 0.00006)
network_train.compile(loss=None,optimizer=optimizer)
network_train.summary()
plot_model(network_train,show_shapes=True, show_layer_names=True, to_file='02 model.png')

print(network_train.metrics_names)

n_iteration=0
network_train.load_weights("saved_weights.h5")


#-----------------START TRAINING-----------------#
triplets = get_batch_random(3)

print("Checking batch width, should be 3 : ",len(triplets))
print("Shapes in the batch A:{0} P:{1} N:{2}".format(triplets[0].shape, triplets[1].shape, triplets[2].shape))
drawTriplets(triplets)

evaluate_every = 25 # interval for evaluating on one-shot tasks
batch_size = 100
n_iter = 1000 # No. of training iterations

print("------------ Training process ------------")

t_start = time.time()
dummy_target = [np.zeros((batch_size,15)) for i in range(3)]
for i in range(1, n_iter+1):
    triplets = get_batch_random(3) # get_batch_hard(200,16,16,FRmodel)
    loss = network_train.train_on_batch(triplets, None)
    n_iteration += 1
    if i % evaluate_every == 0:
        print("[{3}] Time for {0} iterations: {1:.1f} mins, Train Loss: {2}".format(i, (time.time()-t_start)/60.0,loss,n_iteration))

network_train.save_weights("saved_weights.h5")
#-----------------END TRAINING-----------------#


database = {}
for i in y_test:
    database[i] = img_to_encoding(dataPath+i+'_2.jpg',FRmodel)

print("\n------------ Positive match test ------------")
positiveTestsCount = 0
falseNegatives = 0
for j in range(nb_classes):
    for i in range(1, 20, 2):
        positiveTestsCount += 1
        dist, result = verify(dataPath + y_test[j]+'_'+str(i)+'.jpg', y_test[j], database, FRmodel)
        print((dist, result))
        if(result == False):
            falseNegatives += 1

print("\n------------ Negative match test ------------")
negativeTestsCount = 0
falsePositives = 0
for j in range(nb_classes):
    for i in range(1, 20, 2):
        negativeTestsCount += 1
        dist, result = verify(dataPath + y_test[j]+'_'+str(i)+'.jpg', y_test[j-1], database, FRmodel)
        print((dist, result))
        if(result == True):
            falsePositives += 1

print("\n------------ Final results ------------")
totalTestsCount = positiveTestsCount + negativeTestsCount
totalFalseFlags = falseNegatives + falsePositives

print("Positive match test accuracy = ", 100-(falseNegatives/positiveTestsCount)*100, "%")
print("Negative match test accuracy = ", 100-(falsePositives/negativeTestsCount)*100, "%")
print("Total accuracy = ", 100-(totalFalseFlags/totalTestsCount)*100, "%")

