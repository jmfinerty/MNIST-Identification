import numpy as np
from time import time
from csvread import read_norm_mnist
from nnmath import forward_pass, backprop, rand_weights, add_bias
from visualize import show_mnist_multiple


def main():

    trials = int(input("\nTrials:  "))
    hnodes = int(input("H Nodes: "))
    epochs = int(input("Epochs:  "))
    alpha = float(input("Alpha:   "))

    if trials <= 0:
        print("Invalid number of trials!\n")
        exit()
    if hnodes <= 1:
        print("Must have more than 1 hidden node!\n")
        exit()

    print("\nReading data... ", end='', flush=True)
    read_time = time()
    labels, values = read_norm_mnist("train", 1)
    values = add_bias(values, 1)
    print(round(time()-read_time, 4), "s.", sep='')

    # trackers for final summary information
    sum_results = 0
    sum_trial_time = 0
    sum_epoch_time = 0

    print("\nBEGINNING TRIALS.")
    for trial in range(trials):

        print("Trial ", trial+1, ": ", sep='', end='',
              flush=True)  # flush needed for end=''

        trial_start_time = time()

        # Running/building the neural net
        print("Building NN... ", end='', flush=True)
        w_1 = rand_weights(785, hnodes)  # 784+1 for bias term
        w_2 = rand_weights(hnodes, 1)
        w_1[0] = 1  # bias' weight
        for _ in range(epochs):
            epoch_start_time = time()
            for row in range(len(values)):
                x = np.array(values[row]).T  # x column vector
                a_h, a_o = forward_pass(x, w_1, w_2)
                w_1, w_2 = backprop(labels[row], x, w_1, w_2, a_h, a_o, alpha)
            sum_epoch_time += time() - epoch_start_time
        trial_time = time() - trial_start_time
        sum_trial_time += trial_time
        print(round(trial_time, 4), "s. ", end='', sep='', flush=True)

        # Testing the neural net
        print("Testing NN... ", end='', flush=True)
        labels, values = read_norm_mnist("test", 1)
        values = add_bias(values, 1)
        correct_images = []
        incorrect_images = []
        for row in range(len(values)):
            # rounds e.g. .998->1, 9.21e-05->0
            result = round(forward_pass(values[row], w_1, w_2)[1])
            if result == labels[row]:
                # [1:] removes bias for printing images later
                correct_images.append(values[row][1:])
            else:
                incorrect_images.append(values[row][1:])
        result = len(correct_images) / len(labels)
        sum_results += result

        print(round(result, 4), "%.", sep='')

    accuracy = round(100 * (sum_results / trials), 4)
    mean_build_time = round(sum_trial_time / trials, 4)
    mean_epoch_time = round(sum_epoch_time / (trials*epochs), 4)
    print("\n==============================================")
    print(trials, "trials,", hnodes, "hnodes,",
          epochs, "epochs,", alpha, "alpha")
    print("Mean rate of accurate categorization: ", accuracy, '%', sep='')
    print("Mean time to build neural net:        ", mean_build_time, 's', sep='')
    print("Mean time of one epoch:               ", mean_epoch_time, 's', sep='')
    print("==============================================")

    print("\nThere were", len(incorrect_images),
          "miscategorized images in the final trial.")
    view = input("Would you like to view them? [Y/N]: ")
    if view.lower() == "y":
        show_mnist_multiple(incorrect_images)
    print()


main()
