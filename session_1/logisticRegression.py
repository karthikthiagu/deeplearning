# Built-in libraries
import time

# External libraries
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

# Data split
num_training_examples = 500 / 2
num_testing_examples  = 100 / 2

# Train data
def createData(num_examples):
    # Parameters of positive Gaussian
    mean = [0, 0]
    cov  = [[1, 0], [0, 1]]
    # Generate examples from the positive class
    positive_examples = np.random.multivariate_normal(mean = mean, cov = cov, size = num_examples)
    positive_labels   = np.ones(num_examples)
    # Parameters of negative Gaussian
    mean = [4, 4]
    cov  = [[1, 0], [0, 10]]
    # Generate examples from the negative class
    negative_examples = np.random.multivariate_normal(mean = mean, cov = cov, size = num_examples)
    negative_labels   = np.zeros(num_examples)
    
    # Pair examples and labels
    positive = zip(positive_examples, positive_labels)
    negative = zip(negative_examples, negative_labels)
    # Collect them in a single structure
    data = positive + negative
    # Shuffle data
    np.random.shuffle(data)
    # Seperate data into examples and labels; convert them into numpy arrays
    data = zip(*data)
    data = [np.asarray(item) for item in data]
    
    # Return data
    return data

def plotData(data):
    examples, labels = data
    positive_x, positive_y = [], []
    negative_x, negative_y = [], []
    for i in range(len(examples)):
        if labels[i] == 1.0:
            positive_x.append(examples[i][0])
            positive_y.append(examples[i][1])
        else:
            negative_x.append(examples[i][0])
            negative_y.append(examples[i][1])
    #fig, ax = plt.subplots( nrows=1, ncols=1 )
    plt.axis((-5, 9, -5, 9))
    plt.scatter(positive_x, positive_y, color = 'b')
    plt.scatter(negative_x, negative_y, color = 'r')
    plt.show(block = False)    

def plotHyperplane(data, w, b):
	plotData(data)
	time.sleep(4)
	x = np.linspace(-10, 10, 100)
	y = -(w[0]*x + b) / w[1]
	plt.plot(x, y)
	plt.show()

def buildModel(train_data, test_data):
    examples, labels = train_data
    num_examples, dim = examples.shape
    # Data
    x = T.matrix('x')
    y = T.vector('y')
    # Model
    w = theano.shared(value = np.asarray(np.random.normal(0, 5, dim), dtype = theano.config.floatX))
    b = theano.shared(value = np.asarray(0,   dtype = theano.config.floatX))
    proj_x = T.dot(x, w) + b
    p_y_given_x = 1 / (1 + T.exp(-proj_x))
    # Loss
    loss = -1 * T.sum(y * T.log(p_y_given_x) + (1 - y) * T.log(1 - p_y_given_x), axis = 0)
    # Gradients; weight updates
    alpha = 0.1
    gradients = T.grad(loss, [w, b])
    updates = [\
                    (w, w - alpha*gradients[0]),\
                    (b, b - alpha*gradients[1])\
              ]
    # Function to train the model
    train = theano.function([x, y], loss, updates = updates)
    # Function to test the model
    test = theano.function([x, y], [loss, p_y_given_x])
    # Batch size; epochs
    batch_size = 50
    num_epochs = 5
    num_batches = num_examples / batch_size
    # Perform Gradient descent
    for epoch in range(num_epochs):
        ###################  Train model ############################
        loss_per_epoch = []
        for batch in range(num_batches):
            loss_per_batch = train(examples[batch * batch_size : (batch + 1) * batch_size],\
                                   labels[batch * batch_size : (batch + 1) * batch_size])
            loss_per_epoch.append(loss_per_batch)
        print 'Loss after %s epochs = %s' % (str(epoch + 1), str(np.mean(loss_per_epoch)))
        ##################  Test model ##############################
        test_loss, p_y = test(test_data[0], test_data[1])
        pred_y = np.zeros_like(p_y)
        for i in range(p_y.shape[0]):
            if p_y[i] >= 0.5:
                pred_y[i] = 1.0
        accuracy = float(np.sum(pred_y == test_data[1])) / test_data[1].shape[0]
        print 'Test loss = %s, Test accuracy = %s' % (str(test_loss), str(accuracy))
        
    # Return learnt parameters
    return w.get_value(), b.get_value()

if __name__ == '__main__':
    # Create training data
    train_data = createData(num_training_examples)
    print 'shape of training examples = ', train_data[0].shape
    print 'shape of training lables   = ', train_data[1].shape

    # Create test data
    test_data  = createData(num_testing_examples)
    print 'shape of testing examples = ', test_data[0].shape
    print 'shape of testing lables   = ', test_data[1].shape
    
    # Plot training data
    #plotData(train_data)

    # Build Logistic Regression model
    w, b = buildModel(train_data, test_data)

    # Plot Hyperplane obtained
    plotHyperplane(train_data, w, b)

