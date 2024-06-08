from keras.datasets import fashion_mnist
from matplotlib import pyplot
import numpy as np

# Load the Fashion MNIST dataset. This returns two tuples of numpy arrays.
# The first tuple represents the training set and the second tuple the test set.
(train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()

# Print the shape of the training set images
print('X_train: ' + str(train_X.shape))

# Print the shape of the training set labels
print('Y_train: ' + str(train_y.shape))

# Print the shape of the test set images
print('X_test:  '  + str(test_X.shape))

# Print the shape of the test set labels
print('Y_test:  '  + str(test_y.shape))

# PLOT 1
# Define class names for Fashion MNIST
class_names = [
    "T-shirt/top", "Trousers", "Pullover", "Dress", "Coat",
    "Sandals", "Shirt", "Sneakers", "Bag", "Ankle boots"
]

# Define the number of images to display
num = 10

# Select the first 'num' images and labels from the training set
images = train_X[:num]
labels = train_y[:num]

# Define the number of rows and columns for the subplot grid
num_row = 2
num_col = 5

# Create a figure and a grid of subplots
fig, axes = pyplot.subplots(num_row, num_col, figsize=(2*num_col, 2.5*num_row))

# Loop over the range of 'num' and plot each image on the subplot grid
for i in range(num):
    # Determine the current subplot
    ax = axes[i // num_col, i % num_col]

    # Display the image in grayscale
    ax.imshow(images[i], cmap='gray')

    # Set the title of the subplot to the corresponding label and class name
    ax.set_title(f"Label: {labels[i]} ({class_names[labels[i]]})")  # Include text representation

# Adjust the layout so that there is no overlap between subplots
pyplot.tight_layout()

# Display the figure with the subplots
pyplot.show()

# PLOT 2
# Create a dictionary to store one image per label
label_to_image = {}

# Iterate through the dataset and select one image for each label
for i in range(len(class_names)):
    # The current label is the index in the class_names list
    label = i

    # Find the index of the first occurrence of the current label in the training labels
    image_index = (train_y == label).nonzero()[0][0]  # Get the first occurrence of the label

    # Add the corresponding image to the dictionary
    label_to_image[label] = train_X[image_index]

# Create a figure and a grid of subplots
fig, axes = pyplot.subplots(num_row, num_col, figsize=(2*num_col, 2.5*num_row))

# Loop over the labels and images in the dictionary
for i, label in enumerate(label_to_image):
    # Determine the current subplot
    ax = axes[i // num_col, i % num_col]

    # Display the image in grayscale
    ax.imshow(label_to_image[label], cmap='gray')

    # Set the title of the subplot to the corresponding label and class name
    ax.set_title(f"Label: {label} ({class_names[label]})")

    # Hide the axes of the subplot
    ax.axis('off')

# Adjust the layout so that there is no overlap between subplots
pyplot.tight_layout()

# Display the figure with the subplots
pyplot.show()

# PLOT 3
# Count the occurrences of each label in the training set
# np.bincount returns the count of each value in an array of non-negative integers
train_label_counts = np.bincount(train_y)

# Count the occurrences of each label in the testing set
test_label_counts = np.bincount(test_y)

# Create a new figure with specified size
pyplot.figure(figsize=(10, 10))

# Create a subplot for the training set
pyplot.subplot(2, 1, 1)  # This creates a grid of 2 rows and 1 column and selects the first plot for drawing

# Create a bar plot for the training set
# range(len(class_names)) gives the x coordinates, train_label_counts gives the heights of the bars
pyplot.bar(range(len(class_names)), train_label_counts, tick_label=class_names, color='skyblue')

# Set the title, x-label, and y-label for the plot
pyplot.title("Training Set Label Distribution")
pyplot.xlabel("Clothing Category")
pyplot.ylabel("Frequency")

# Create a subplot for the testing set
# This selects the second plot in the grid for drawing
pyplot.subplot(2, 1, 2)

# Create a bar plot for the testing set
# Same as above but for the test set
pyplot.bar(range(len(class_names)), test_label_counts, tick_label=class_names, color='salmon')

# Set the title, x-label, and y-label for the plot
pyplot.title("Testing Set Label Distribution")
pyplot.xlabel("Clothing Category")
pyplot.ylabel("Frequency")

# Adjust the layout so that there is no overlap between subplots
pyplot.tight_layout()

# Display the figure with the subplots
pyplot.show()
