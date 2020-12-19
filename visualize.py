import matplotlib.pyplot as plt


# given row of 784 grayscale values, shows image
def show_mnist(values):
    image = []
    i = 0
    while i+28 < len(values):
        image.append(values[i:i+28])
        i += 28
    plt.imshow(image, cmap='gray')
    plt.show()


# given list of lists of 784 grayscale values,
# shows them all in one image
def show_mnist_multiple(images_values):

    fig = plt.figure(figsize=(8, 8))

    # format list of 1x784 images into list of 28x28 images
    images = []
    for values in images_values:
        image = []
        i = 0
        while i+28 < len(values):
            image.append(values[i:i+28])
            i += 28
        images.append(image)

    # determine dimensions of display grid
    # by picking largest pair of factors of # of images
    # e.g. 20 images: display images in 4x5 plot
    factor_pair = []
    # only need to check for factors up to sqrt
    for i in range(1, int(len(images_values) ** .5) + 1):
        if len(images_values) % i == 0:
            # saved pair is last discovered pair of factors
            factor_pair = [i, int(len(images_values) / i)]

    # build and display grid of images
    rows = factor_pair[0]
    cols = factor_pair[1]
    for i in range(1, (cols*rows)+1):
        fig.add_subplot(rows, cols, i)
        plt.imshow(images[i-1], cmap="gray")

    plt.show()
