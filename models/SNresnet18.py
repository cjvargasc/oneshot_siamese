import torch
import torchvision
import torchvision.utils
import torch.nn as nn


class SiameseNetwork(nn.Module):
    """ Loads the pretrained Resnet18 model. Set mode for fine-tuning """

    def __init__(self, mode=1, pretrained=True):
        super(SiameseNetwork, self).__init__()

        self.model_conv = torchvision.models.resnet18(pretrained=pretrained)

        self.net_parameters = []  # List of parameters requiring grad

        #for name, child in self.model_conv.named_children():
        #    if name != "fc":
        #        for name2, params in child.named_parameters():
        #            params.requires_grad = False
        #    else:
        #        for name2, params in child.named_parameters():
        #            params.requires_grad = True
        #            self.net_parameters.append(params)

        for param in self.model_conv.parameters():
            param.requires_grad = False

        self.model_conv = self.model_conv.cuda()

        self.out_last = self.model_conv.fc.out_features

        self.extraL = nn.Linear(self.out_last, 1)
        for param in self.extraL.parameters():
            param.requires_grad = True
            self.net_parameters.append(param)

        self.extraL = self.extraL.cuda()


    def forward_once(self, x):
        output = self.model_conv(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # compute l1 distance
        diff = torch.abs(output1 - output2)
        # score the similarity between the 2 encodings
        scores = self.extraL(diff)

        return output1, output2, scores