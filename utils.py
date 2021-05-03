import torch
import os
import numpy as np
import cv2


def convert_points_to_image(samp_pts, H, W):
	"""
	Inputs:-
	samp_pts: b, 1, k, 2
	"""

	b, _, K, _ = samp_pts.shape
	# Convert pytorch -> numpy.
	samp_pts = samp_pts.data.cpu().numpy().reshape(b, K, 2)
	samp_pts = (samp_pts + 1.) / 2.
	samp_pts = samp_pts * np.array([float(W-1), float(H-1)]).reshape(1, 1, 2)
	return samp_pts.astype(np.int32)


def convert_points_to_torch(pts, H, W, device="cuda:0"):
    """
    Inputs:-
    pts: k, 2 (W, H)
    """

    samp_pts = torch.from_numpy(pts.astype(np.float32))
    samp_pts[:, 0] = (samp_pts[:, 0] * 2. / (W-1)) - 1.
    samp_pts[:, 1] = (samp_pts[:, 1] * 2. / (H-1)) - 1.
    samp_pts = samp_pts.view(1, 1, -1, 2)
    samp_pts = samp_pts.float().to(device)
    return samp_pts


def visualize_keypoints(images1, images2, output1, output2, mask, out_dir="./", base_name="im"):
	images1 = cv2.cvtColor(images1, cv2.COLOR_GRAY2RGB)
	images2 = cv2.cvtColor(images2, cv2.COLOR_GRAY2RGB)
	
	im = np.concatenate([images1, images2], axis=1)	
	color = [0, 0, 1]

	for k1, l1 in enumerate(output1):
		x1, y1 = l1
		cv2.circle(im, (x1, y1), 2, color, -1)
		for k2, l2 in enumerate(output2):
			x2, y2 = l2
			cv2.circle(im, (x2+images1.shape[1], y2), 2, color, -1)
			if mask[k1, k2] == 1:
				cv2.line(im, (x1, y1), (x2+images1.shape[1], y2), (0, 1, 0), 1)

	cv2.imwrite(os.path.join(out_dir, "{}.jpg".format(base_name)), (im*255).astype(np.uint8))