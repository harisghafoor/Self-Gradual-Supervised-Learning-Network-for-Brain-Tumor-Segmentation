import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        self.enc1 = conv_block(1, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.enc5 = conv_block(512, 1024)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        self.dec4 = conv_block(1024, 512)
        self.dec3 = conv_block(512, 256)
        self.dec2 = conv_block(256, 128)
        self.dec1 = conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, 2, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        print("Shape after enc1:", x1.shape)
        p1 = self.pool(x1)
        print("Shape after pool1:", p1.shape)
        
        x2 = self.enc2(p1)
        print("Shape after enc2:", x2.shape)
        p2 = self.pool(x2)
        print("Shape after pool2:", p2.shape)
        
        x3 = self.enc3(p2)
        print("Shape after enc3:", x3.shape)
        p3 = self.pool(x3)
        print("Shape after pool3:", p3.shape)
        
        x4 = self.enc4(p3)
        print("Shape after enc4:", x4.shape)
        p4 = self.pool(x4)
        print("Shape after pool4:", p4.shape)
        
        x5 = self.enc5(p4)
        print("Shape after enc5:", x5.shape)
        
        # Decoder
        up4 = self.upconv4(x5)
        print("Shape after upconv4:", up4.shape)
        up4 = self.crop_and_concat(x4, up4)
        print("Shape after crop_and_concat4:", up4.shape)
        d4 = self.dec4(up4)
        print("Shape after dec4:", d4.shape)
        
        up3 = self.upconv3(d4)
        print("Shape after upconv3:", up3.shape)
        up3 = self.crop_and_concat(x3, up3)
        print("Shape after crop_and_concat3:", up3.shape)
        d3 = self.dec3(up3)
        print("Shape after dec3:", d3.shape)
        
        up2 = self.upconv2(d3)
        print("Shape after upconv2:", up2.shape)
        up2 = self.crop_and_concat(x2, up2)
        print("Shape after crop_and_concat2:", up2.shape)
        d2 = self.dec2(up2)
        print("Shape after dec2:", d2.shape)
        
        up1 = self.upconv1(d2)
        print("Shape after upconv1:", up1.shape)
        up1 = self.crop_and_concat(x1, up1)
        print("Shape after crop_and_concat1:", up1.shape)
        d1 = self.dec1(up1)
        print("Shape after dec1:", d1.shape)
        
        out = self.final_conv(d1)
        print("Shape after final_conv:", out.shape)
        
        return out

    def crop_and_concat(self, enc_features, upconv_output):
        _, _, H, W = upconv_output.shape
        enc_features = self.center_crop(enc_features, H, W)
        return torch.cat([enc_features, upconv_output], dim=1)

    def center_crop(self, layer, max_height, max_width):
        _, _, h, w = layer.size()
        start_h = (h - max_height) // 2
        start_w = (w - max_width) // 2
        return layer[:, :, start_h:start_h + max_height, start_w:start_w + max_width]

# # Checkpoint shape reproduction
# input_tensor = torch.randn(16, 1, 572, 572)
# model = UNet()
# output = model(input_tensor)
# print("Output shape:", output.shape)

if __name__ == "__main__":
    device = torch.device("mps")
    x = torch.randn((16, 1, 572, 572)).to(device)
    f = UNet().to(device)
    main_output = f(x)
    print("Main Input Shape:", x.shape)
    print("Main Output Shape:", main_output.shape)
#     print("Second Last Decoder Output Shape:", ds_outputs[0].shape)
#     print("Third Last Decoder Output Shape:", ds_outputs[1].shape)