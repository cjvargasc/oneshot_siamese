import torch
import torchvision
import torchvision.utils
import torch.nn as nn


class SiameseNetwork(nn.Module):
    def __init__(self, lastLayer=False, pretrained=True):

        super(SiameseNetwork, self).__init__()

        self.lastLayer = lastLayer
        self.net_parameters = []  # list of parameters to be optimized

        self.model_conv = torchvision.models.alexnet(pretrained=pretrained)

        if pretrained:
            # freeze all parameters in the model
            for param in self.model_conv.parameters():
                param.requires_grad = False

            #  Unfreeze model last layer
            self.out_last = self.model_conv.classifier[6].out_features
            # reset last layer?
            # self.model_conv.classifier[6] = nn.Linear(4096, self.out_last)
            for param in self.model_conv.classifier[6].parameters():
                param.requires_grad = True
                self.net_parameters.append(param)
                #torch.nn.init.xavier_uniform(param)
        else:
            for param in self.model_conv.parameters():
                param.requires_grad = True
                self.net_parameters.append(param)
                #torch.nn.init.xavier_uniform(param)

        if self.lastLayer:
            self.out_last = self.model_conv.classifier[6].out_features

            self.extraL = nn.Linear(self.out_last, 1)
            for param in self.extraL.parameters():
                param.requires_grad = True
                self.net_parameters.append(param)
                #torch.nn.init.xavier_uniform(param)

    def forward_once(self, x):
        output = self.model_conv(x)
        #if self.lastLayer:
        #    output = self.extraL(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        if self.lastLayer:
            # compute l1 distance (similarity) between the 2 encodings
            diff = torch.abs(output1 - output2)
            scores = self.extraL(diff)
            return scores
        else:
            return output1, output2
