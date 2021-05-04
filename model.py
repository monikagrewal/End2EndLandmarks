import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import numpy as np

from unet import *
from utils import *


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


class DescMatchingModule(nn.Module):
	"""
	DescMatchingModule
	"""
	def __init__(self, in_channels, out_channels):
		super(DescMatchingModule, self).__init__()
		self.fc = nn.Linear(in_channels, out_channels)
		self.apply(weight_init)


	def forward(self, out1, out2):
		b, c, h1, w1 = out1.size()
		b, c, h2, w2 = out2.size()

		out1 = out1.view(b, c, h1*w1).permute(0, 2, 1).view(b, h1*w1, 1, c)
		out2 = out2.view(b, c, h2*w2).permute(0, 2, 1).view(b, 1, h2*w2, c)

		# all possible descriptor pairs
		out = out1 * out2
		out = out.contiguous().view(-1, c)

		out = self.fc(out)

		# normalize input features
		dn1 = torch.norm(out1, p=2, dim=3) # Compute the norm.
		out1 = out1.div(1e-6 + torch.unsqueeze(dn1, 3)) # Divide by norm to normalize.
		dn2 = torch.norm(out2, p=2, dim=3) # Compute the norm.
		out2 = out2.div(1e-6 + torch.unsqueeze(dn2, 3)) # Divide by norm to normalize.

		out_norm = torch.norm(out1 - out2, p=2, dim=3)
		return out, out_norm


class Net(nn.Module):
	"""
	What follows is awesomeness redefined
	"""
	def __init__(self, in_channels=1, out_channels=2, batchnorm=False, threeD=False, depth=4, width=16,\
				 device="cuda:0", k=512, scale_factor=8):
		super(Net, self).__init__()
		self.device = device
		self.k = k
		self.scale_factor = scale_factor
		self.CNN_branch = UNet(depth=depth, width=width, growth_rate=2, in_channels=in_channels, out_channels=1)
		feature_channels = self.CNN_branch.feature_channels

		self.desc_matching_layer = DescMatchingModule(feature_channels, out_channels)


	def forward(self, x1, x2):
		k = self.k
		scale_factor = self.scale_factor

		# landmark detection and description
		heatmaps1, features1 = self.CNN_branch(x1)
		heatmaps2, features2 = self.CNN_branch(x2)

		# sampling top k landmark locations and descriptors
		landmarks1, landmark_probs1, desc1 = self.sampling_layer(heatmaps1, features1, is_training=True)
		landmarks2, landmark_probs2, desc2 = self.sampling_layer(heatmaps2, features2, is_training=True)

		# descriptor matching probabilities and descriptor norms
		desc_pairs_score, desc_pairs_norm = self.desc_matching_layer(desc1, desc2)

		return landmark_probs1, landmark_probs2, landmarks1, landmarks2, desc_pairs_score, desc_pairs_norm


	def predict(self, x1, x2, deformation=None, conf_thresh=0.01, k=None):
		if k is None:
			k = self.k
		scale_factor = self.scale_factor
		b, _, H, W = x1.shape
		# landmark detection and description
		heatmaps1, features1 = self.CNN_branch(x1)
		heatmaps2, features2 = self.CNN_branch(x2)

		# sampling top k landmark locations and descriptors
		pts1, _, desc1 = self.sampling_layer(heatmaps1, features1, conf_thresh=conf_thresh, is_training=False)
		pts2, _, desc2 = self.sampling_layer(heatmaps2, features2, conf_thresh=conf_thresh, is_training=False)

		# descriptor matching probabilities and descriptor norms
		desc_pairs_score, desc_pairs_norm = self.desc_matching_layer(desc1, desc2)

		# post processing
		landmarks1 = convert_points_to_image(pts1, H, W)
		landmarks2 = convert_points_to_image(pts2, H, W)

		b, k1, _ = landmarks1.shape
		_, k2, _ = landmarks2.shape

		# two-way (bruteforce) matching
		desc_pairs_score = F.softmax(desc_pairs_score, dim=1)[:,1].view(b, k1, k2)
		desc_pairs_score = desc_pairs_score.detach().to("cpu").numpy()
		desc_pairs_norm = desc_pairs_norm.detach().to("cpu").numpy()
		matches = list()
		for i in range(b):
			pairs_score = desc_pairs_score[i]
			pairs_norm = desc_pairs_norm[i]
			
			match_cols = np.zeros((k1, k2))
			match_cols[np.argmax(pairs_score, axis=0), np.arange(k2)] = 1
			match_rows = np.zeros((k1, k2))
			match_rows[np.arange(k1), np.argmax(pairs_score, axis=1)] = 1
			match = match_rows * match_cols

			match_cols = np.zeros((k1, k2))
			match_cols[np.argmin(pairs_norm, axis=0), np.arange(k2)] = 1
			match_rows = np.zeros((k1, k2))
			match_rows[np.arange(k1), np.argmin(pairs_norm, axis=1)] = 1
			match = match * match_rows * match_cols
			
			matches.append(match)

		matches = np.array(matches)

		if deformation is not None:
			deformation = deformation.permute(0, 3, 1, 2)  #b, 2, h, w
			pts1_projected = F.grid_sample(deformation, pts2) #b, 2, 1, k
			pts1_projected = pts1_projected.permute(0, 2, 3, 1) #b, 1, k, 2
			landmarks1_projected = convert_points_to_image(pts1_projected, H, W)
			return landmarks1, landmarks2, matches, landmarks1_projected
		else:
			return landmarks1, landmarks2, matches


	def sampling_layer(self, heatmaps, features, conf_thresh=0.000001, is_training=True):
		k = self.k
		scale_factor = self.scale_factor
		device = self.device

		b, _, H, W = heatmaps.shape
		heatmaps = torch.sigmoid(heatmaps)
		
		"""
		Convert pytorch -> numpy after maxpooling and unpooling
		This is faster way of sampling while ensuring sparsity
		One could alternatively apply non-maximum suppresion (NMS)
		"""
		if is_training:
			heatmaps1, indices = F.max_pool2d(heatmaps, (scale_factor, scale_factor), stride=(scale_factor, scale_factor), return_indices=True)
			heatmaps1 = F.max_unpool2d(heatmaps1, indices, (scale_factor, scale_factor))
			heatmaps1 = heatmaps1.to("cpu").detach().numpy().reshape(b, H, W)
		else:
			heatmaps1 = heatmaps.to("cpu").detach().numpy().reshape(b, H, W)

		# border mask, optional
		border = 10
		border_mask = np.zeros_like(heatmaps1)
		border_mask[:, border : H - border, border : W - border] = 1.
		heatmaps1 = heatmaps1 * border_mask

		all_pts= []
		for heatmap in heatmaps1:
			xs, ys = np.where(heatmap >= conf_thresh) # get landmark locations above conf_thresh
			if is_training:
				if len(xs) < k:
					xs, ys = np.where(heatmap >= 0.0)
			pts = np.zeros((len(xs), 3))
			pts[:, 0] = ys
			pts[:, 1] = xs
			pts[:, 2] = heatmap[xs, ys]
			inds = np.argsort(pts[:, 2])
			pts = pts[inds[::-1], :] # sort by probablity scores
			pts = pts[:k, :2] #take top k

			# Interpolate into descriptor map using 2D point locations.
			samp_pts = convert_points_to_torch(pts, H, W, device=device)
			all_pts.append(samp_pts)

		all_pts = torch.cat(all_pts, dim=0)
		pts_score = F.grid_sample(heatmaps, all_pts)  #b, 1, 1, k
		pts_score = pts_score.permute(0, 3, 1, 2).view(b, -1)
		desc = [F.grid_sample(desc, all_pts) for desc in features]
		desc = torch.cat(desc, dim=1)
		return all_pts, pts_score, desc


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
	pass
