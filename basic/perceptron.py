import random

def show_learning(w):
    print('w0 =', '%5.2f' % w[0], ', w1 =', '%5.2f' % w[1], ', w2 =', '%5.2f' % w[2])

#Define variables needed to control training process
random.seed(7) #To make repeatable
LEARNING_RATE = 0.1
index_list = [0, 1, 2, 3] #Used to randomize order


#Training examples
x_train = [(1.0, -1.0, -1.0), (1.0, -1.0, 1.0), 
            (1.0, 1.0, -1.0), (1.0, 1.0, 1.0)]
y_train = [1.0, 1.0, 1.0, -1.0]

#Define perceptrion weights
w = [0.2, -0.6, 0.25]

#Prints inital weights
show_learning(w)

#First element in vector x must be 1.
#Length of w and x must be n+1 for neuron n inputs.
def compute_output(w, x):
    z = 0.0
    for i in range(len(w)):
        z += x[i] * w[i] #compute sum of weighted inputs.
        if z < 0.0: #Apply sign functuin.
            return -1
        else:
            return 1

#Perceptrion training loop

all_correct = True
while not all_correct:
    all_correct = True
    random.shuffle(index_list) #Random order
    for i in index_list:
        x = x_train[i]
        y = y_train[i]
        p_out = compute_output(w, x) #Perceptron function
        if y != p_out: #Updates weights when wrong
            for j in range(0, len(w)):
                w[j] += (y * LEARNING_RATE * x[j])
            all_correct = False
            show_learning(w) #Shows updated weights
print("End of code")
