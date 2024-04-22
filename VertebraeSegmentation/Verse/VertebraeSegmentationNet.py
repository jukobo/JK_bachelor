import torch
import torch.nn as nn

    
def DoubleConv(in_channels, feature_maps, dropout):
        double_conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels,out_channels=feature_maps,kernel_size=3, stride = 1, padding = 1),
            nn.BatchNorm3d(feature_maps),
            nn.Dropout3d(p=dropout),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_channels=feature_maps,out_channels=feature_maps,kernel_size=3, stride = 1, padding = 1),
            nn.BatchNorm3d(feature_maps),
            nn.Dropout3d(p=dropout),
            nn.LeakyReLU(inplace=True),
        )
        return double_conv


class Unet3D(nn.Module):
    def __init__(self, dropout):
        super(Unet3D, self).__init__()

        #Double convolutions
        self.conv_down1 = DoubleConv(2,64,dropout)
        self.conv_down2 = DoubleConv(64,64,dropout)
        self.conv_down3 = DoubleConv(64,64,dropout)
        self.conv_down4 = DoubleConv(64,64,dropout)
        self.bottom = DoubleConv(64,64,dropout)
        self.conv_up4 = DoubleConv(128,64,dropout)
        self.conv_up3 = DoubleConv(128,64,dropout)
        self.conv_up2 = DoubleConv(128,64,dropout)
        self.conv_up1 = DoubleConv(128,64,dropout)

        #Average pooling
        self.avgpool = nn.AvgPool3d(kernel_size=2,stride=2)

        #Linear upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')

        #Output
        self.output = nn.Conv3d(in_channels=64,out_channels=1,kernel_size=3, stride = 1, padding = 1)

        #Initialisation strategy
        #nn.init.normal_(self.output.weight, std=0.001)

    def forward(self, image):
        #Contracting
        layer1_skip = self.conv_down1(image)
        x = self.avgpool(layer1_skip)

        layer2_skip = self.conv_down2(x)
        x = self.avgpool(layer2_skip)

        layer3_skip = self.conv_down3(x)
        x = self.avgpool(layer3_skip)
        
        layer4_skip = self.conv_down4(x)
        x = self.avgpool(layer4_skip)
        
        #Parallel
        x = self.bottom(x)
        
        #Expanding
        x = self.upsample(x)
        x = self.conv_up4(torch.cat((layer4_skip,x),dim=1))
        
        x = self.upsample(x)
        x = self.conv_up3(torch.cat((layer3_skip,x),dim=1))
        
        x = self.upsample(x)
        x = self.conv_up2(torch.cat((layer2_skip,x),dim=1))
        
        x = self.upsample(x)
        x = self.conv_up1(torch.cat((layer1_skip,x),dim=1))
        
        output = self.output(x)
        return output
    
class VertebraeSegmentationNet(nn.Module):
    def __init__(self, dropout):
        super(VertebraeSegmentationNet, self).__init__()

        #Double convolutions
        self.conv_down1 = DoubleConv(2,64,dropout)
        self.conv_down2 = DoubleConv(64,64,dropout)
        self.conv_down3 = DoubleConv(64,64,dropout)
        self.conv_down4 = DoubleConv(64,64,dropout)
        self.bottom = DoubleConv(64,64,dropout)
        self.conv_up4 = DoubleConv(128,64,dropout)
        self.conv_up3 = DoubleConv(128,64,dropout)
        self.conv_up2 = DoubleConv(128,64,dropout)
        self.conv_up1 = DoubleConv(128,64,dropout)

        #Average pooling
        self.avgpool = nn.AvgPool3d(kernel_size=2,stride=2)

        #Linear upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')

        #Output
        self.output = nn.Conv3d(in_channels=64,out_channels=1,kernel_size=3, stride = 1, padding = 1)

        #Initialisation strategy
        #nn.init.normal_(self.output.weight, std=0.001)

    def forward(self, image):
        #Contracting
        layer1_skip = self.conv_down1(image)
        x = self.avgpool(layer1_skip)

        layer2_skip = self.conv_down2(x)
        x = self.avgpool(layer2_skip)

        layer3_skip = self.conv_down3(x)
        x = self.avgpool(layer3_skip)
        
        layer4_skip = self.conv_down4(x)
        x = self.avgpool(layer4_skip)
        
        #Parallel
        x = self.bottom(x)
        
        #Expanding
        x = self.upsample(x)
        x = self.conv_up4(torch.cat((layer4_skip,x),dim=1))
        
        x = self.upsample(x)
        x = self.conv_up3(torch.cat((layer3_skip,x),dim=1))
        
        x = self.upsample(x)
        x = self.conv_up2(torch.cat((layer2_skip,x),dim=1))
        
        x = self.upsample(x)
        x = self.conv_up1(torch.cat((layer1_skip,x),dim=1))
        
        output = self.output(x)
        return output
    
    #def expanding(self, image):
if __name__ == "__main__":
    image = torch.rand((1,2,96,96,128))
    model = Unet3D(0.0)
    print(model)
    #Call model
    print(model(image).shape)


