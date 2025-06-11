import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import keras
import tensorflow as tf

#directly import dataset module from keras
mnist= tf.keras.datasets.mnist 
#x is pixel data and y is classification data
(x_train, y_train), (x_test,y_test) = mnist.load_data()
 
#normalize the data
x_train= tf.keras.utils.normalize(x_train,axis=1)
x_test= tf.keras.utils.normalize(x_test, axis=1)

# model=tf.keras.models.Sequential()
# #now flatten the image
# model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# #add dense layers
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='softmax'))

# #compile the model afterward`s`
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# #now fit/train the model
# model.fit(x_train,y_train, epochs=3)

# model.save('handwritten.keras')


model = tf.keras.models.load_model('handwritten.keras')


image_number=1
while os.path.isfile(f"test/digit{image_number}.png"):
    try:
        img = cv2.imread(f"test/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction=model.predict(img)
        print(f"This digit is prob {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("error")
    finally:
        image_number+=1
# #evaluate loss and accuracy, we need low loss and high accuracy
# loss, accuracy = model.evaluate(x_test,y_test)

# print(loss)
# print(accuracy)