import torch
import torch.nn.functional as F
import numpy as np


def get_labels(pts1, pts2, deformation, device="cuda:0"):
	"""
	pts1 = b, 1, k, 2
	deformation = b, h, w, 2
	"""
	k = pts1.shape[2]
	b, h, w, _ = deformation.shape

	"""
	--- threshold on number of pixels to decide match ---
	since Pytorch coordinates extend from -1 to 1, 1 pixel ~= 2/h pytorch coordinates unit
	so, thresh = 2/h means that a landmark is matching if it's projection on another image
	lies within 1 pixel distance (which is quite strict).
	One may try making thresh more lenient, which will allow for more (less spatially accurate) landmark matches 
	"""
	thresh = torch.tensor(2./float(h), device=device)

	# map landmarks in image 2 on image 1
	deformation = deformation.permute(0, 3, 1, 2)  #b, 2, h, w
	pts1_projected = F.grid_sample(deformation, pts2) #b, 2, 1, k

	pts1_projected = pts1_projected.permute(0, 2, 3, 1) #b, 1, k, 2
	pts1 = pts1.view(b, -1, 1, 2)
	cell_distances = torch.norm(pts1 - pts1_projected, dim=3)

	# two-way (bruteforce) matching
	min_cell_distances_row = torch.min(cell_distances, dim=1)[0].view(b, 1, -1)
	min_cell_distances_col = torch.min(cell_distances, dim=2)[0].view(b, -1, 1)
	s1 = torch.eq(cell_distances, min_cell_distances_row)
	s2 = torch.eq(cell_distances, min_cell_distances_col)
	s = s1 * s2 * torch.ge(thresh, cell_distances)  #b, k, k
	s = s.float()

	indices = torch.nonzero(s)
	gt1 = torch.zeros(b, s.shape[1], dtype=torch.float).to(device)
	gt2 = torch.zeros(b, s.shape[2], dtype=torch.float).to(device)
	gt1[indices[:, 0], indices[:, 1]] = 1.
	gt2[indices[:, 0], indices[:, 2]] = 1.

	return gt1, gt2, s


def custom_loss(landmark_probs1, landmark_probs2, desc_pairs_score, desc_pairs_norm, gt1, gt2, match_target, k, device="cuda:0"):
	# LandmarkProbabilityLoss Image 1
	landmark_probs1_lossa = torch.mean(torch.tensor(1.).to(device) - torch.sum(landmark_probs1, dim=(1)) / torch.tensor(float(k)).to(device))
	landmark_probs1_lossb = F.binary_cross_entropy(landmark_probs1, gt1)
	landmark_probs1_loss = landmark_probs1_lossa + landmark_probs1_lossb
	
	# LandmarkProbabilityLoss Image 2
	landmark_probs2_lossa = torch.mean(torch.tensor(1.).to(device) - torch.sum(landmark_probs2, dim=(1)) / torch.tensor(float(k)).to(device))
	landmark_probs2_lossb =	F.binary_cross_entropy(landmark_probs2, gt2)
	landmark_probs2_loss = landmark_probs2_lossa + landmark_probs2_lossa

	# descriptor loss
	b, k1, k2 = match_target.shape
	wt = float(k) / float(k)**2
	desc_loss1 = F.cross_entropy(desc_pairs_score, match_target.long().view(-1),
	 weight=torch.tensor([wt, 1 - wt]).to(device))

	Npos = match_target.sum()
	Nneg = b*k1*k2 - Npos
	pos_loss = torch.sum(match_target * torch.max(torch.zeros_like(desc_pairs_norm).to(device), desc_pairs_norm - 0.1)) / (2*Npos + 1e-6)
	neg_loss = torch.sum((1.0 - match_target) * torch.max(torch.zeros_like(desc_pairs_norm).to(device), 1.0 - desc_pairs_norm)) / (2*Nneg + 1e-6)
	desc_loss2 = pos_loss + neg_loss
	desc_loss = desc_loss1 + desc_loss2

	# total loss
	loss = landmark_probs1_loss + landmark_probs2_loss + desc_loss

	return loss