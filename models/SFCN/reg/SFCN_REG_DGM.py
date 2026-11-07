import torch
import torch.nn as nn

class SFCN_Regression(nn.Module):
    def __init__(self, channel_number=[32, 64, 128, 256, 256, 64], prediction_range=(5, 100), dropout=True):
        super(SFCN_Regression, self).__init__()
        self.prediction_range = prediction_range
        n_layer = len(channel_number)
        
        # Backbone
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = channel_number[i-1]
            out_channel = channel_number[i]
            if i < n_layer-1:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=True,
                                                                  kernel_size=3,
                                                                  padding=1))
            else:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=False,
                                                                  kernel_size=1,
                                                                  padding=0))
        #regression
        self.classifier = nn.Sequential()
        avg_shape = [2, 2, 2]
        self.classifier.add_module('average_pool', nn.AvgPool3d(avg_shape))
        if dropout is True:
            self.classifier.add_module('dropout', nn.Dropout(0.2))
        i = n_layer
        in_channel = channel_number[-1]
        self.classifier.add_module('conv_%d' % i,
                                   nn.Conv3d(in_channel, 1, padding=0, kernel_size=1))
        self.classifier.add_module('flatten', nn.Flatten())  # output

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
        return layer

    def forward(self, x):
        x_f = self.feature_extractor(x)
        x = self.classifier(x_f)
        
        #if self.prediction_range is not None:
            #lower, upper = self.prediction_range
            #x = torch.clamp(x, min=lower, max=upper)
        return x
