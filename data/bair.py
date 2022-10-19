from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import bisect
import numpy as np
import albumentations
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from torch.utils.data import Dataset, ConcatDataset
import cv2
import os

class ConcatDatasetWithIndex(ConcatDataset):
    """Modified from original pytorch code to return dataset idx"""
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx


class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def get_mask_GT_Gaussian(self, image_A, image_B):
        img_a = cv2.imread(image_A)
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        img_a = cv2.GaussianBlur(img_a, (5, 5), 0)
        img_b = cv2.imread(image_B)
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
        img_b = cv2.GaussianBlur(img_b, (5, 5), 0)
        frameDelta = cv2.absdiff(img_a, img_b)
        thresh = cv2.threshold(frameDelta, 15, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=1)
        mask_GT = convex_hull_image(thresh)
        # thresh = cv2.resize(thresh, (16,16))
        mask_GT = (mask_GT).astype(np.float32)

        return mask_GT, thresh.astype(np.float32)

    def get_mask_GT(self, image_A, image_B):
        img_a = cv2.imread(image_A)
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        img_b = cv2.imread(image_B)
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
        
        frameDelta = img_a + img_b
        frameDelta = np.clip(frameDelta, 0, 255)
        thresh = cv2.threshold(frameDelta, 1, 255, cv2.THRESH_BINARY)[1]
        mask_GT = (thresh/255).astype(np.float32)
        # print(np.mean(mask_GT))
        return mask_GT

    def __getitem__(self, i):
        example = dict()
        a_path = self.labels["file_path_"][i]
        example["image"] = self.preprocess_image(a_path)
        a_frame_idx = int(a_path.split('/')[-1].split('.')[0])
        b_path = ''
        for s in a_path.split('/')[:-1]:
            b_path += s + '/'
        b_path += str(a_frame_idx + 1) + '.png'

        c_path = ''
        for s in a_path.split('/')[:-1]:
            c_path += s + '/'
        c_path += str(a_frame_idx + 2) + '.png'

        if not os.path.exists(b_path):
            b_path = a_path
        if not os.path.exists(c_path):
            c_path = a_path

        example['image_B'] = self.preprocess_image(b_path)

        # if mask_GT exists:
        # if a_path.split('/')[-4] == "test_gt":
        #     a_mask_path = b_path.replace("test_gt", "test_mask_gt")
        #     b_mask_path = c_path.replace("test_gt", "test_mask_gt")
        # else:
        #     a_mask_path = b_path.replace("train", "train_mask_gt")
        #     b_mask_path = c_path.replace("train", "train_mask_gt")
        # example['mask_GT'] = self.get_mask_GT(a_mask_path, b_mask_path)

        # print(a_path, a_mask_path, b_mask_path)
        # example['mask_GT'], example['mask_thresh'] = self.get_mask_GT_Gaussian(a_path, b_path)
        
        # example['mask_GT'] = None
        # example['mask_thresh'] = None
        example['img_path'] = self.labels["file_path_"][i]

        # first_frame_path = ''
        # for s in self.labels["file_path_"][i].split('/')[:-1]:
        #     first_frame_path += s + '/'
        # first_frame_path += str(0) + '.png'
        # example['image_anchor'] = self.preprocess_image(first_frame_path)

        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class NumpyPaths(ImagePaths):
    def preprocess_image(self, image_path):
        image = np.load(image_path).squeeze(0)  # 3 x 1024 x 1024
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(image, mode="RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
