import torch
import torchvision
import torchvision.utils
import torch.nn as nn


class SiameseNetwork(nn.Module):
    def __init__(self, mode=1, pretrained=True):

        super(SiameseNetwork, self).__init__()

        self.model_conv = torchvision.models.alexnet(pretrained=pretrained)
        #self.model_conv = torchvision.models.resnet18(pretrained=pretrained)

        self.net_parameters = []

        #if mode == 1:  # Fine-tune last layer
        ##### Fine tune alexnet:
        # https://discuss.pytorch.org/t/does-anyone-have-the-fine-tuning-alexnet-code/4111/6
        # https://medium.com/@14prakash/almost-any-image-classification-problem-using-pytorch-i-am-in-love-with-pytorch-26c7aa979ec4
        #print("mode=1 -- fine-tuning last layer")

        for param in self.model_conv.parameters():
            param.requires_grad = False
        ''''''
        for param in self.model_conv.classifier[6].parameters():
            param.requires_grad = True
            self.net_parameters.append(param)

        self.model_conv = self.model_conv.cuda()

        ''' '''
        self.out_last = self.model_conv.classifier[6].out_features

        self.extraL = nn.Linear(self.out_last, 1)
        for param in self.extraL.parameters():
            param.requires_grad = True
            self.net_parameters.append(param)

        self.extraL = self.extraL.cuda()


    def forward_once(self, x):
        output = self.model_conv(x)
        #output = self.extraL(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # compute l1 distance
        diff = torch.abs(output1 - output2)
        # score the similarity between the 2 encodings
        scores = self.extraL(diff)

        return output1, output2, scores
