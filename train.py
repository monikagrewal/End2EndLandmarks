import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import os, argparse
import json
import numpy as np
from tqdm import tqdm

from model import Net
from data import CelebADataset
from loss import custom_loss, get_labels
from utils import visualize_keypoints


def train(model, train_dataloader, optimizer, device="cuda:0", out_dir="./"):
	train_loss = 0.
	nbatches = len(train_dataloader)
	pbar = tqdm(enumerate(train_dataloader), desc="training", total=nbatches, unit="batches")
	for batch_no, (images1, images2, deformations) in pbar:
		images1, images2, deformations = images1.to(device), images2.to(device), deformations.to(device)

		optimizer.zero_grad()
		landmark_probs1, landmark_probs2, keypoints1, keypoints2, desc_pairs_score, desc_pairs_norm = model(images1, images2)
		gt1, gt2, match_target = get_labels(keypoints1, keypoints2, deformations, device=device)

		loss = custom_loss(landmark_probs1, landmark_probs2, desc_pairs_score, desc_pairs_norm, gt1, gt2, match_target, model.k, device=device)
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		pbar.set_postfix({'Loss': loss.item()})

	train_loss = train_loss / float(nbatches)
	return train_loss


def validation(model, val_dataloader, device="cuda:0", out_dir="./"):
	val_loss = 0.
	nbatches = len(val_dataloader)
	pbar = tqdm(enumerate(val_dataloader), desc="validation", total=nbatches, unit="batches")
	for batch_no, (images1, images2, deformations) in pbar:
		images1, images2, deformations = images1.to(device), images2.to(device), deformations.to(device)

		with torch.no_grad():
			landmark_probs1, landmark_probs2, keypoints1, keypoints2, desc_pairs_score, desc_pairs_norm = model(images1, images2)
			gt1, gt2, match_target = get_labels(keypoints1, keypoints2, deformations, device=device)

			loss = custom_loss(landmark_probs1, landmark_probs2, desc_pairs_score, desc_pairs_norm, gt1, gt2, match_target, model.k, device=device)
			val_loss += loss.item()
		pbar.set_postfix({'Loss': loss.item()})

		if batch_no%100 == 0:
			output1, output2, output3 = model.predict(images1, images2)
			images1 = images1.to("cpu").numpy()
			images2 = images2.to("cpu").numpy()

			for i in range(images1.shape[0]):
				im1 = images1[i,0,:,:]
				im2 = images2[i,0,:,:]
				out1 = output1[i]
				out2 = output2[i]
				mask = output3[i]
				visualize_keypoints(im1.copy(), im2.copy(), out1, out2, mask, out_dir=out_dir, base_name="iter_{}_{}".format(batch_no, i))

	val_loss = val_loss / float(nbatches)
	return val_loss


def parse_input_arguments(parser):
	run_params = parser.parse_args()
	run_params = vars(run_params)

	if run_params["data_dir"] is None:
		run_params["data_dir"] = "/export/scratch3/grewal/Data/CelebA"
		# raise IOError("data_dir argument missing.")

	out_dir = run_params["out_dir"]
	os.makedirs(out_dir, exist_ok=True)

	json.dump(run_params, open(os.path.join(out_dir, "run_parameters.json"), "w"))
	return run_params


def main():
	parser = argparse.ArgumentParser(description='Train Landmark Detection')
	parser.add_argument("-data_dir", help="root directory of CelebA dataset", type=str, default=None)
	parser.add_argument("-device", help="cuda number", type=int, default=0)
	parser.add_argument("-out_dir", help="output directory", default="./runs/run1")
	parser.add_argument("-image_size", help="image size", type=int, default=128)
	parser.add_argument("-depth", help="network depth", type=int, default=4)
	parser.add_argument("-width", help="network width", type=int, default=16)
	parser.add_argument("-nepochs", help="number of epochs", type=int, default=5)
	parser.add_argument("-lr", help="learning rate", type=float, default=0.001)
	parser.add_argument("-batchsize", help="batchsize", type=int, default=8)
	parser.add_argument("-k", help="number of landmarks to sample", type=int, default=256)
	parser.add_argument("-scale_factor", help="sparsity in landmarks", type=int, default=8)
	run_params = parse_input_arguments(parser)

	if torch.cuda.is_available():
		device = "cuda:{}".format(run_params["device"])
	else:
		device = "cpu"
	out_dir, nepochs, lr, batchsize = run_params["out_dir"], run_params["nepochs"], run_params["lr"], run_params["batchsize"]
	image_size = run_params["image_size"]
	depth, width = run_params["depth"], run_params["width"]
	
	out_dir_train = os.path.join(out_dir, "train")
	out_dir_val = os.path.join(out_dir, "val")
	out_dir_wts = os.path.join(out_dir, "weights")
	os.makedirs(out_dir_train, exist_ok=True)
	os.makedirs(out_dir_val, exist_ok=True)
	os.makedirs(out_dir_wts, exist_ok=True)

	root_dir = run_params["data_dir"]
	train_dataset = CelebADataset(root_dir, image_size=image_size, split='train')
	validation_dataset = CelebADataset(root_dir, image_size=image_size, split='valid')
	print("training data: {}, validation data: {}".format(len(train_dataset), len(validation_dataset)))
	train_dataloader = DataLoader(train_dataset, batch_size=batchsize, num_workers=3, shuffle=True)
	validation_dataloader = DataLoader(validation_dataset, batch_size=batchsize, num_workers=3, shuffle=False)
	
	model = Net(in_channels=1, device=device, depth=depth, width=width, k=run_params["k"], scale_factor=run_params["scale_factor"])
	model.to(device)
	optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

	best_val_loss = np.inf
	for epoch in range(0, nepochs):
		print("Epoch {}".format(epoch))
		# training
		model.train()
		train_loss = train(model, train_dataloader, optimizer, device=device, out_dir=out_dir_train)

		# validation
		model.eval()
		val_loss = validation(model, validation_dataloader, device=device, out_dir=out_dir_val)

		print("Training Loss: {}, Validation Loss: {}\n".format(train_loss, val_loss))
		# saving model
		if val_loss<=best_val_loss:
			weights = {"model": model.state_dict()}
			torch.save(weights, os.path.join(out_dir_wts, "model_weights.pth"))

if __name__ == '__main__':
	main()


