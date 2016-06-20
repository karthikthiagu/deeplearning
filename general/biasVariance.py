import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import time

fig = plt.figure()
plot_window_1 = fig.add_subplot(121)
plot_window_2 = fig.add_subplot(122)

def learnAndPlot(plot_window, bins, num_examples):
    # Learn the distribution
    # num_runs is the number of times the model (histogram) will be learnt
    num_runs = 50
    for run in range(num_runs):
        # Draw training examples from the true distribution, a Gaussian, in this case
        data = np.random.normal(loc = 0.0, scale = 1, size = 1000)
        # Get the histogram corresponding to this configuration (bins) and plot the result
        plot_window.hist(data, bins = bins, color = 'w', normed = True)

    # Plot the original distribution, a Gaussian
    mu = 0  # mean
    var = 1 # variance
    sigma = np.sqrt(var) # std-deviation
    x = np.linspace(-3, 3, 100) # take 100 equally spaced points between -3 and 3
    p_x = mlab.normpdf(x, mu, sigma)    # get the normalized probabilites
    plot_window.plot(x, p_x)    # plot

# Number of training examples to get a histogram
num_examples = 1000
# High variance : have a lot of bins => every run of the experiment will try to overfit to the particular sample of 1000 points 
bins = np.linspace(-3, 3, 100)
learnAndPlot(plot_window_1, bins, num_examples)
# But note that the shape of the histogram more or less resembles the true distribution, i.e., low bias
# High bias : have very few bins, only two in our case => model underfits, doesn't capture the nature of the distribution
bins = np.linspace(-3, 3, 2)
learnAndPlot(plot_window_2, bins, num_examples)
# Note that the shape of the histogram is a flat line, i.e., it has predicted a uniform distribution in the place of the Gaussian.
# This is underfitting. As the model complexity (bin size) is low, we are not able to approximate the original distribution.
# The variance is low, as expected. When 1000 samples are drawn from a 0-centered Gaussian, approximately half will fall on either
# side of zero. This will happen every time 1000 samples are drawn. Thus the variance is low.
plt.show()
