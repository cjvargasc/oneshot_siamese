import torch
import torchvision
import torchvision.utils
import torch.nn as nn


class SiameseNetwork(nn.Module):
    def __init__(self, pretrained=True):

        super(SiameseNetwork, self).__init__()

        self.net_parameters = [] # list of parameters to be optimized

        self.model_conv = torchvision.models.alexnet(pretrained=pretrained)
        #self.model_conv = torchvision.models.resnet18(pretrained=pretrained)
        #self.model_conv = torchvision.models.resnet50(pretrained=pretrained)
        #self.model_conv = torchvision.models.resnet101(pretrained=pretrained)
        #self.model_conv = torchvision.models.vgg16(pretrained=pretrained)
        #self.model_conv = torchvision.models.vgg16_bn(pretrained=pretrained)
        #self.model_conv = torchvision.models.squeezenet1_0(pretrained=pretrained)
        #self.model_conv = torchvision.models.densenet161(pretrained=pretrained)
        #self.model_conv = torchvision.models.inception_v3(pretrained=pretrained)

        # freeze all parameters in the model
        for param in self.model_conv.parameters():
            param.requires_grad = False

        #  Unfreeze model last layer
        ''' vgg / alexnet '''
        ### Next line resets the layer
        self.out_last = self.model_conv.classifier[6].out_features
        self.model_conv.classifier[6] = nn.Linear(4096, self.out_last)
        for param in self.model_conv.classifier[6].parameters():
            param.requires_grad = True
            self.net_parameters.append(param)


        ''' resnet18 
        for param in self.model_conv.fc.parameters():
            param.requires_grad = True
            self.net_parameters.append(param)
        self.out_last = self.model_conv.fc.out_features
        '''

        ''' squezednet 
        for param in self.model_conv.classifier[1].parameters():
            param.requires_grad = True
            self.net_parameters.append(param)
        self.out_last = self.model_conv.classifier[1].out_features
        '''

        ''' densenet 
        for param in self.model_conv.classifier.parameters():
            param.requires_grad = True
            self.net_parameters.append(param)
        self.out_last = self.model_conv.classifier.out_features
        '''

        ''' Inception v3  
        for param in self.model_conv.AuxLogits.fc.parameters():
            param.requires_grad = True
            self.net_parameters.append(param)
        for param in self.model_conv.fc .parameters():
            param.requires_grad = True
            self.net_parameters.append(param)
        self.out_last = self.model_conv.fc.out_features
        '''

        # Set architecture last layer (trained dist)
        #self.extraL = nn.Linear(self.out_last, 1)
        #for param in self.extraL.parameters():
        #    param.requires_grad = True
        #    self.net_parameters.append(param)

        #self.extraL = self.extraL.cuda()

    def forward_once(self, x):
        output = self.model_conv(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # compute l1 distance
        #diff = torch.abs(output1 - output2)
        # score the similarity between the 2 encodings
        #scores = self.extraL(diff)

        return output1, output2#, scores
