import random
from models.layers import *



class VGGSNN(nn.Module):
    def __init__(self):
        super(VGGSNN, self).__init__()
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        #pool = APLayer(2)
        self.features = nn.Sequential(
            Layer(2,64,3,1,1),
            Layer(64,128,3,1,1),
            pool,
            Layer(128,256,3,1,1),
            Layer(256,256,3,1,1),
            pool,
            Layer(256,512,3,1,1),
            Layer(512,512,3,1,1),
            pool,
            Layer(512,512,3,1,1),
            Layer(512,512,3,1,1),
            pool,
        )
        W = int(48/2/2/2/2)
        # self.T = 4
        self.classifier = SeqToANNContainer(nn.Linear(512*W*W,10))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        # input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x

class VGGSNNwoAP(nn.Module):
    def __init__(self):
        super(VGGSNNwoAP, self).__init__()
        self.features = nn.Sequential(
            Layer(2,64,3,1,1),
            Layer(64,128,3,2,1),
            Layer(128,256,3,1,1),
            Layer(256,256,3,2,1),
            Layer(256,512,3,1,1),
            Layer(512,512,3,2,1),
            Layer(512,512,3,1,1),
            Layer(512,512,3,2,1),
        )
        W = int(48/2/2/2/2)
        # self.T = 4
        self.classifier = SeqToANNContainer(nn.Linear(512*W*W,10))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        # input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x



if __name__ == '__main__':
    model = VGGSNNwoAP()
    