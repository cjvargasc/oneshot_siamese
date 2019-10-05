
class Config:

    ## Dataset
    training_dir = "/home/mmv/Documents/3.datasets/openlogo/1.preproc/training/"
    testing_dir = "/home/mmv/Documents/3.datasets/openlogo/1.preproc/testing/"
    #training_dir = "/home/camilo/siamese_test/data/logos/training/"
    #testing_dir = "/home/camilo/siamese_test/data/logos/testing/"
    #training_dir = "/home/mmv/Documents/3.datasets/Omniglot/proc/training/"
    #testing_dir = "/home/mmv/Documents/3.datasets/Omniglot/proc/testing/"


    #full_test_dir = "/home/mmv/Documents/3.datasets/openlogo/full/"
    full_test_dir = "/home/mmv/Documents/3.datasets/Omniglot/proc/"

    # Alexnet 224,224 or 300,300
    im_w = 224
    im_h = 224

    ## Model params
    #model = "alexnet"
    #model = "resnet"
    #model = "koch"
    model = "tests"
    mode = 1  # 1: fine tune last layer for alexnet (check models)
    pretrained = True

    train_batch_size = 64
    train_number_epochs = 200
    lrate = 0.0005

    ## Model save/load path
    best_model_path = "testmodel.pt"
    model_path = "testmodel_last.pt"

