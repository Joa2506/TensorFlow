import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.utils as plot_model


#Contiunation of nnr but with bigger datasets 

X = tf.range(-100, 100, 4)

print(X)

#Make labels for datasets

y = X + 10

print(y)

#Visualize data

#plt.plot(X, y)
#plt.show()

plt.scatter(X, y)
#plt.show()

###The 3 sets

#Training set - the model learns from this data, 70 - 80% of the total data you have available

#Validation set - The model gets tuned on this data, which is 10 - 15% of data available

#Test set - the model gets evaluated on this data to test what it has learned, also typically 10 - 15 % of total data available

#Check the length of how many examples one has
print(len(X))

# Split the data into train and test set

X_train = X[:40] #First 40 are training samples (80% of the data)
y_train = y[:40]


X_test = X[40:] # last 10 testing samples (20% of the data)
y_test = y[40:]

print(len(X_train), len(y_train), len(X_test), len(X_test))

#Evaluating visualize, visualize, visualize

###Visualizing the data, got data in training and test set

plt.figure(figsize=(10, 7))
#Plot training data in blue
plt.scatter(X_train, y_train, c="b", label="Training data")
#Plot testing data in green
plt.scatter(X_test, y_test, c="g", label="Testing data")

plt.legend()
#plt.show()

#Let's have a look at how to build a neural network for the data

#1. Create model
# model = keras.Sequential([
#     keras.layers.Dense(1)
# ])

# #2. Compile model
# model.compile(
#     loss=keras.losses.mae,
#     optimizer=keras.optimizers.SGD(),
#     metrices=["mae"])

#3. Fit model
#model.fit(X_train, y_train, epochs=100)

###Visualizing the model###

tf.random.set_seed(42)
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=[1], name="input_layer"),# X[0] = -100, this is just a number. That's why the input shape is just a number [1]. Tensor is just a scalar
    keras.layers.Dense(1, name="output_layer")
], name="model1")#Dense means fully connected layer

model.compile(
    loss=keras.losses.mae,
    optimizer=keras.optimizers.SGD(), #Stochastic gradient descent
    metrics=["mae"])

#Total params - total number of params in the model
#Trainable params - these are parameters (patterns) the model can update as it trains
#Non-trainable params - parameters can't be updated as it trains. Freezed parameters. (Typical when one brings in already trained pattern or parameters from other models during training)
model.fit(X_train, y_train, epochs=100, verbose=0)
model.summary()
plot_model.plot_model(model=model, show_shapes=True)

###Visualize models prediction###

#To visualize predictions it's a good idea to plot the agains the ground truth labels.

#Often in the form of 'y_test' or 'y_true' versus 'y_pred' (ground truth of models predictions)

#Make predictions

y_pred = model.predict(X_test)
#print(y_test)
print(y_pred)

#A plotting function

#Plots training data, test data and compares predictions to ground truth
def plotting_predictions(training_data=X_train,
                         train_labels=y_train,
                         test_data=X_test,
                         test_labels=y_test,
                         prediction=y_pred):
    figure = plt.figure(figsize=(10, 7))
    #Plot training data in blue
    figure = plt.scatter(training_data, train_labels, c="b", label="Training data")
    #Plot testing data in green
    figure = plt.scatter(test_data, test_labels, c="g", label="Test data")
    #Plot prediction data in red
    figure = plt.scatter(test_data, prediction, c="r", label="Predictions")
    #Show legend
    figure = plt.legend()
    figure = plt.show()
#plotting_predictions()


###Evaluating model's prediction with regression evaluation metrics

#Dependent on the problem there are different evaluation metrics to evaluate model's performance

##Two main ones
#1 MAE - Mean absolute error, "On average, how wrong is each of my model's predictions". Great for starting with any regression problem

#2 MSE - Mean square error, "Square the average errors". When lareger errors are more significant than smaller errors

#3 Huber - HUBEr, Takes the combination of MSE and MAE, less sensitive to outliers than MSE

#Evaluate model on test set
model.evaluate(X_test, y_test)

#Calculate mean absolute error

mae = keras.losses.mean_absolute_error(y_test, y_pred)
print(tf.constant(y_pred))
print(tf.constant(y_test))
#Different shape need to reshape

tf.squeeze(y_pred)

mae = keras.losses.mean_absolute_error(y_true=y_test, y_pred=tf.squeeze(y_pred)) 
print("MAE:", mae)
mse = keras.losses.mean_squared_error(y_true=y_test, y_pred=tf.squeeze(y_pred))
print("MSE:", mse)
huber = keras.losses.huber(y_true=y_test, y_pred=tf.squeeze(y_pred))
print("Huber:", huber)