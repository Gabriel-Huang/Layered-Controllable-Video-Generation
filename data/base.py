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
    def __init__(self, paths, size=None, random_crop=False, labels=None, skip_frames = 0):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)
        self.skip_frames = skip_frames + 1

        self.rescaler = albumentations.Resize(self.size[0], self.size[1])
        self.preprocessor = albumentations.Compose([self.rescaler])

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

    def preprocess_mask(self, image_path):
        image = Image.open(image_path).convert('L')
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/255).astype(np.float32)
        image = np.expand_dims(image, axis=0)
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
        return mask_GT

    def __getitem__(self, i):
        example = dict()
        a_path = self.labels["file_path_"][i]
        example["image"] = self.preprocess_image(a_path)
        a_frame_idx = int(a_path.split('/')[-1].split('.')[0])
        b_path = ''
        for s in a_path.split('/')[:-1]:
            b_path += s + '/'
        b_path += f'{(a_frame_idx + self.skip_frames):05}.png'

        if not os.path.exists(b_path):
            b_path = a_path
        # print('a', a_path, 'b', b_path)
        example['image_B'] = self.preprocess_image(b_path)


        # mask = self.preprocess_mask(mask_path)
        # example['mask'] = mask
        # example['img_path'] = self.labels["file_path_"][i]

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
