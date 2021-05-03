import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init


def conv_block(in_ch=1, out_ch=1, threeD=True):
	if threeD:
		layer = nn.Sequential(nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
								nn.BatchNorm3d(out_ch),
								nn.ReLU()
								)
	else:
		layer = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
								nn.BatchNorm2d(out_ch),
								nn.ReLU()
								)		
	return layer


def deconv_block(in_ch=1, out_ch=1, scale_factor=2, threeD=True):
	if threeD:
		layer = nn.Sequential(nn.ConvTranspose3d(in_ch, out_ch, kernel_size=scale_factor, stride=scale_factor),
								nn.BatchNorm3d(out_ch),
								nn.ReLU()
								)
	else:
		layer = nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=scale_factor, stride=scale_factor),
								nn.BatchNorm2d(out_ch),
								nn.ReLU()
								)		
	return layer


def Unet_DoubleConvBlock(in_ch=1, out_ch=1, threeD=True):
	layer = nn.Sequential(conv_block(in_ch=in_ch, out_ch=out_ch, threeD=threeD),
							conv_block(in_ch=out_ch, out_ch=out_ch, threeD=threeD)
							)
	return layer


class UNet(nn.Module):
	"""
	Implementation of U-Net:
	Paper: U-Net: Convolutional Networks for Biomedical Image Segmentation
	Full text available at: https://arxiv.org/abs/1505.04597
	"""
	def __init__(self, depth=4, width=64, growth_rate=2, in_channels=1, out_channels=2, threeD=False):
		super(UNet, self).__init__()
		self.depth = depth
		self.out_channels = [width*(growth_rate**i) for i in range(self.depth+1)]

		# Downsampling Path Layers
		self.downblocks = nn.ModuleList()
		current_in_channels = in_channels
		for i in range(self.depth+1):
			self.downblocks.append(Unet_DoubleConvBlock(current_in_channels, self.out_channels[i], threeD=threeD))
			current_in_channels = self.out_channels[i]

		self.feature_channels = current_in_channels + self.out_channels[i-1]
		# Upsampling Path Layers
		self.deconvblocks = nn.ModuleList()
		self.upblocks = nn.ModuleList()
		for i in range(self.depth):
			self.deconvblocks.append(deconv_block(current_in_channels, self.out_channels[-2 - i], threeD=threeD))
			self.upblocks.append(Unet_DoubleConvBlock(current_in_channels, self.out_channels[-2 - i], threeD=threeD))
			current_in_channels = self.out_channels[-2 - i]

		if threeD:
			self.last_layer = nn.Conv3d(current_in_channels, out_channels, kernel_size=1)
			self.downsample = nn.MaxPool3d(2)
		else:
			self.last_layer = nn.Conv2d(current_in_channels, out_channels, kernel_size=1)
			self.downsample = nn.MaxPool2d(2)			

		# Initialization
		self.apply(self.weight_init)


	def forward(self, x):
		# Downsampling Path
		out = x
		down_features_list = list()
		for i in range(self.depth):
			out = self.downblocks[i](out)
			down_features_list.append(out)
			out = self.downsample(out)

		# bottleneck
		out = self.downblocks[-1](out)
		features = [down_features_list[-1], out]

		# Upsampling Path
		for i in range(self.depth):
			out = self.deconvblocks[i](out)
			down_features = down_features_list[-1 - i]
			out = torch.cat([down_features, out], dim=1)
			out = self.upblocks[i](out)

		out = self.last_layer(out)	

		return out, features


	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
			torch.nn.init.kaiming_normal_(m.weight.data)
			if m.bias is not None:
				m.bias.data.fill_(0.0)
		if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
			m.weight.data.fill_(1.0)
			m.bias.data.fill_(0.0)
		if isinstance(m, nn.Linear):
			torch.nn.init.kaiming_normal_(m.weight.data)
			if m.bias is not None:
				m.bias.data.fill_(0.0)



if __name__ == '__main__':
	model = UNet(depth=4, width=16).cuda()
	inputs = torch.rand(1, 1, 128, 128).cuda()
	output, features = model(inputs)
	print (output.shape, features[0].shape)