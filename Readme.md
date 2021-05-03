# Self-supervised End-to-End Landmark Detection and Matching
Code for the paper *[An end-to-end deep learning approach for landmark detection and matching in medical images](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11313/2549302/An-end-to-end-deep-learning-approach-for-landmark-detection/10.1117/12.2549302.short?SSO=1)*. The full text of the paper is available at [https://arxiv.org/pdf/2001.07434v1.pdf](https://arxiv.org/pdf/2001.07434v1.pdf).

Note, the implementation for generating ground truth labels (function `get_labels` in `loss.py`) required for self-supervised training of landmark matches is similar to (and inspired by) the method mentioned in *[End-to-end learning of keypoint detector and descriptor for pose invariant 3D matching](https://openaccess.thecvf.com/content_cvpr_2018/papers/Georgakis_End-to-End_Learning_of_CVPR_2018_paper.pdf)*.

### Usage
Due to restrictions in sharing the medical data used in the paper, the repository is modified to train on CelebA dataset. To train on CelebA dataset, download CelebA dataset from the following link: [http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
Clone the repository and install dependencies using `requirements.txt`. In the root folder of the repository, run the following command on terminal:

```python
python train.py -data_dir <path to CelebA dataset>
```

If you find the code useful, please cite the following paper:

```
@inproceedings{grewal2020end,
  title={An end-to-end deep learning approach for landmark detection and matching in medical images},
  author={Grewal, Monika and Deist, Timo M and Wiersma, Jan and Bosman, Peter AN and Alderliesten, Tanja},
  booktitle={Medical Imaging 2020: Image Processing},
  volume={11313},
  pages={1131328},
  year={2020},
  organization={International Society for Optics and Photonics}
}
```
