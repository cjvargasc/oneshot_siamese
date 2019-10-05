
import os


def rename():

    directory = "/home/mmv/Documents/3.datasets/Omniglot/proc/"

    alph_cont = 0

    for alph_dir in os.listdir(directory):

        for char_dir in os.listdir(directory + alph_dir + "/"):

            os.rename(directory + alph_dir + "/" + char_dir, directory + alph_dir + "/" + str(alph_cont) + "_" + char_dir)

        alph_cont += 1


if __name__ == "__main__":
    rename()
