#Convolutional Neural Networks
# we separated the images into training data set and test data set
# we have separated all the picutres into 2 categories i.e Cats and Dogs

#this means all the data is split so we donrr need the train_test_split

#We dont need to do Data Preprocessing since it is already done our stuff is neatly orgazined already


#Part-1

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


#initialize the CNN
classifier=Sequential()

# Step 1 -->Convolution --> that is feature maps go thru it and then on apply Rectifier to get the conolution layer

classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu' ))

#.add() parameter explain--> since previously we passed arrays
#incase of images we pass it using the Convolution2D method 32,3,3-->features=32 of size 3*3 and input_shape is to make all the images of same size same
#before pooling we have to put the rectifier function that is relu




#Step 2 --> Pooling  --> kill the twists and get the basic features irrespective of orientation

classifier.add(MaxPooling2D(pool_size=(2,2)  ))
#the parameter we sent i.e pool_size gives the pool size frame



#to make the model more sharp i will be adding one more convolution layer
#you can kee adding more to make it more precise and more slow

classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))






#Step 3-->Flattening --> we make the pooled shit into a vector --> which will act as input to the ANN Nueron

classifier.add(Flatten())

#Step 4--> Full conncection-->One hidden layer
classifier.add(Dense(units=128,activation='relu'))

#one final output layer
classifier.add(Dense(units=1,activation='sigmoid'))

#COMPILING THE SHIT
#Optamizer self explanatory
#loss --> how to deal with loss

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])



#PART-2 FITTING THE CNN TO THE IMAGE

#We will augment our data-> it means we will look at the same image in different orientations
#By looking at different orientations we will give more material to learn
#which will provide more accurate material --> USE CODE FROM KERS
from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

trainingset = train_datagen.flow_from_directory(
        'training_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

testset = test_datagen.flow_from_directory(
        'test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        trainingset,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=testset,
        validation_steps=2000)


#with here you are done training and test the data set Now what if we want to predict some new image check out below !

# from keras.preprocessing import image as image_utils
# import numpy as np
#
# test_image = image_utils.load_img('path-of-the-image', target_size=(64, 64))
# test_image = image_utils.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis=0)

result = classifier.predict_on_batch(test_image)