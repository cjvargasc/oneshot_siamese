
class Config:

    ## Dataset
    training_dir = "/home/mmv/Documents/3.datasets/openlogo/preproc/1/training/"
    testing_dir = "/home/mmv/Documents/3.datasets/openlogo/preproc/1/testing/"

    # Alexnet 224,224 or 300,300
    im_w = 224
    im_h = 224

    ## Model params
    model = "alexnet"
    #model = "resnet"
    #model = "dense"
    #model = "inception"
    #model = "vgg"
    #model = "squeeze"
    #model = "koch"
    #model = "tests"
    pretrained = False
    distanceLayer = False  # defines if the last layer uses a distance metric or a neuron output
    bceLoss = False  # If true the feature vectors f(xi) are substracted

    train_batch_size = 64
    train_number_epochs = 200
    lrate = 0.0005

    ## Model save/load path
    best_model_path = "testmodel"
    model_path = "testmodel_last"

