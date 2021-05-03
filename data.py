import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os, glob
import numpy as np
import pandas as pd
from skimage.transform import resize
import cv2


def normalize(image):
	epsilon = 1e-6
	image = (image - np.min(image)) / float(np.max(image) - np.min(image) + epsilon)
	return image


def to_tensor_shape(image):
	# bring channel axis along first dimension (C, H, W in Pytorch)
	if len(image.shape)==3:
		image = image.transpose(2, 0, 1)
	elif len(image.shape)==2:
		image = np.expand_dims(image, axis=0)
	else:
		raise ValueError("Unknown image type")
	return image


def create_affine_matrix(rotation=0, scale=1, shear=0, center=np.array([0, 0])):
	"""
	Input: rotation angles in degrees
	"""
	theta = rotation * np.pi/180
	affine_matrix = np.array([[scale * np.cos(theta), -np.sin(theta + shear), 0],
				[np.sin(theta + shear), scale * np.cos(theta), 0],
				[0, 0, 1]])

	center = center.reshape(-1, 1)
	center_homogenous = np.array([center[0], center[1], 1]).reshape(-1, 1)
	center_rotated = np.dot(affine_matrix, center_homogenous)

	affine_matrix[:2, 2] = center.flatten() - center_rotated.flatten()[:2]
	return affine_matrix.T


def generate_random_2dgaussian(h, w, sigma_h=None, sigma_w=None):
	if sigma_h is None:
		sigma_h = h//8

	if sigma_w is None:
		sigma_w = w//8

	H, W = np.meshgrid(np.linspace(0, h, h), np.linspace(0, w, w), indexing="ij")

	center_h, center_w = torch.randint(h//10, h - h//10, (1, 1)).item(), torch.randint(w//10, w - w//10, (1, 1)).item()
	sigma_h, sigma_w = torch.randint(sigma_h//2, sigma_h, (1, 1)).item(), torch.randint(sigma_w//2, sigma_w, (1, 1)).item()
	mag_h, mag_w = torch.randint(-4, 4, (1, 1)).item() / 20., torch.randint(-4, 4, (1, 1)).item() / 20.

	if mag_h == 0.:
		mag_h = 0.1
	if mag_w == 0.:
		mag_w = 0.1

	g_h = mag_h * np.exp(-((H - center_h)**2 / (2.0 * sigma_h**2)))
	g_w = mag_w * np.exp(-((W - center_w)**2 / (2.0 * sigma_w**2)))
		
	return g_h.reshape(-1), g_w.reshape(-1)


def generate_deformation_grid(image1):
	"""
	Generates a random deformation field, applies it to the input image and returns deformaed image and deformation field.
	
	Inputs:
	image1 = Channels * Height * Width

	Outputs:
	deformation = Height * Width * 2

	"""
	shape = (image1.shape[1], image1.shape[2])
	x, y = np.meshgrid(np.linspace(-1, 1, shape[0]), np.linspace(-1, 1, shape[1]), indexing="ij")
	indices = np.array([np.reshape(x, -1), np.reshape(y, -1), np.ones(shape[0]*shape[1])]).T  #shape N, 3

	choices = ["rotation", "elastic", "scale", "shear"]
	idx = torch.randint(len(choices), (1, 1)).item()
	random_choice = choices[idx]

	if random_choice=="rotation":
		param = (-45, 45)
		angle = torch.randint(param[0], param[1], (1, 1)).item()
		M = create_affine_matrix(rotation=angle)
		indices = np.dot(indices, M)

	elif random_choice=="scale":
		param = (0.8, 1.2)
		scale = torch.randint(int(param[0]*100), int(param[1]*100), (1, 1)).item() / 100.
		M = create_affine_matrix(scale=scale)
		indices = np.dot(indices, M)

	elif random_choice=="shear":
		param = (-20, 20)
		shear = torch.randint(param[0], param[1], (1, 1)).item() * (np.pi/180.)
		M = create_affine_matrix(shear=shear)
		indices = np.dot(indices, M)

	elif random_choice=="elastic":
		dx, dy = generate_random_2dgaussian(shape[0], shape[1])	

		indices[:, 0] += dx
		indices[:, 1] += dy 

	# normalized grid for pytorch
	indices = indices[:, :2].reshape(shape[0], shape[1], 2)
	indices = indices.transpose(1, 0, 2)

	return indices


class CelebADataset(Dataset):
	"""
	Wrapper Dataset class of CelebA dataset.
	Requires CelebA dataset pre-downloaded in root_dir
	"""

	def __init__(self, root_dir, image_size=128, split='train'):
		super(CelebADataset, self).__init__()
		split_map = {"train": 0, "valid": 1, "test": 2, "all": None}
		self.root = root_dir
		self.base_folder = "celeba"

		info = pd.read_csv(os.path.join(self.root, self.base_folder, "list_eval_partition.txt"))
		if split_map.get(split, None) is not None:
			info = info.loc[info.partition==split_map[split]]
		else:
			raise Warning("Not able to understand the provided split str, so using all data. \
					valid split str are: {}".format(list(split_map.keys())))
		self.imlist = info.image_id.to_list()
		self.image_size = image_size


	def __len__(self):
		return len(self.imlist)


	def __getitem__(self, idx):
		sample = os.path.join(self.root, self.base_folder, "img_align_celeba", self.imlist[idx])
		image1 = cv2.imread(sample, 0)
		c = 1

		image1 = normalize(image1)
		image1 = resize(image1, (self.image_size, self.image_size), mode='constant')
		image1 = to_tensor_shape(image1)
		
		deformation = generate_deformation_grid(image1)
		image2 = F.grid_sample(torch.tensor(image1).view(1, c, self.image_size, self.image_size),
		 torch.tensor(deformation).view(1, self.image_size, self.image_size, 2), mode="nearest")
		image2 = image2.numpy().reshape(c, self.image_size, self.image_size)

		return image1.astype(np.float32), image2.astype(np.float32), deformation.astype(np.float32)




