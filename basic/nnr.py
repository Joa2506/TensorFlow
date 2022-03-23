#This is a program made to go through some of the basics of Neural network regression

#Import Tensorflow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
print(tf.__version__)


#Creating features
X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

#Creating labels
y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

#Visualize
plt.scatter(X, y)
#plt.show()

##Input and output shapes
house_info = tf.constant(["bedroom", "bathroom", "garage"])
house_price = tf.constant([939700])
tf.print(house_info)
tf.print(house_price)

input_shape = X[0].shape
output_shape = y[0].shape

tf.print(input_shape)
tf.print(output_shape) 

#Turn numpy arrays into tensors with dtype float32

X = tf.constant(X, dtype=tf.float32)
y = tf.constant(y, dtype=tf.float32)

tf.print(X)
tf.print(y)

input_shape = X[0].shape
output_shape = y[0].shape

tf.print(input_shape)
tf.print(output_shape)



##Steps in modelling with tensorflow
 
#1. Create model - define input and output layers as well as hidden layers of a deep learning model

#2. Compile a model - define loss function (Tells our model how wrong it is) and optimizer (Tells the model how to improve its learning) and evaluation metrics (what we can use to interpret the performance of the model)

#3. Fitting a model - letting the model try to find patterns between X & y(features and labels)

####################################################################################################################

##Let's try

#Set random seed

tf.random.set_seed(42) #The answer to the ultimate question of life, the universe and everything

#1 Create a model using sequential API

model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    keras.layers.Dense(1) #Take one number and predict another number
])

#2 Compile model
model.compile(loss=keras.losses.mae, #mean absolute error
            optimizer=keras.optimizers.SGD(), #stochastic gradient descent
            metrics=["mae"])

#3 Fit model

tf.print(model.fit(X, y, epochs=10)) #Look at X and y and try to fix the pattern. You have 5 opportunities

#Try and make a prediction on our model

tf.print(model.predict([17.0]))

#Improve model, by altering the steps taken to create model

#1 Creating model add more layers and increase number of hidden units(neurons) within each og the hidden layers, change activation function

#2 Compiling model, here might change the optimization function

#3 Fitting model. Fit a model for more epochs (leave training for longer or more data) 

#1
model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(1) #Take one number and predict another number
])
#2
model.compile(loss=keras.losses.mae, #mean absolute error
            optimizer=keras.optimizers.Adam(lr=0.01), #Changed from SGD to Adam
            metrics=["mae"])

#3
tf.print(model.fit(X, y, epochs=100))
tf.print(model.predict([17.0]))

#Common ways to improe a dl model

#Add layers

#Increase hidden units

#Change activation function

#Change optimization function

#Learning rate tweaking (lr very important in many learning settings)

#NB! More is not necessarily better 
