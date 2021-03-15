import os
import cv2
import math
import time
import numpy as np
from PIL import Image
import scipy.io as scio
import random
import torch
from torch._six import container_abcs
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.image import get_affine_transform, gaussian_radius, draw_msra_gaussian


BASEDIR = os.path.dirname(os.path.abspath(__file__))
CFGPATH = '{}/cfg'.format(BASEDIR)


class CameraInfo():
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale


def drawGaussian(img, pt, score, sigma=1):
    """Draw 2d gaussian on input image.
    Parameters
    ----------
    img: torch.Tensor
        A tensor with shape: `(3, H, W)`.
    pt: list or tuple
        A point: (x, y).
    sigma: int
        Sigma of gaussian distribution.
    Returns
    -------
    torch.Tensor
        A tensor with shape: `(3, H, W)`.
    """
    # img = to_numpy(img)
    tmp_img = np.zeros([img.shape[0], img.shape[1]], dtype=np.float32)
    tmpSize = 3 * sigma
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - tmpSize), int(pt[1] - tmpSize)]
    br = [int(pt[0] + tmpSize + 1), int(pt[1] + tmpSize + 1)]

    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 2 * tmpSize + 1
    x = np.arange(0, size, 1, float)
    # print('x:', x.shape)
    y = x[:, np.newaxis]
    # print('x:', x.shape)
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    
    # print('g:', g.shape)
    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)) * score
    g = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    # g = np.concatenate([g[..., np.newaxis], np.zeros([g.shape[0], g.shape[1], 2], dtype=np.float32)], axis=-1)

    tmp_img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g
    # img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g
    img += tmp_img

    return img


class SuctionNetDataset(Dataset):
    def __init__(self, data_root, label_root, camera='kinect', split='train', input_size=(480, 480), adapt_radius=True):
        # assert(num_points<=50000)
        self.data_root = data_root
        self.label_root = label_root
        self.camera = camera
        self.split = split
        self.dim = input_size
        self.adapt_radius = adapt_radius
        self.data_rng = np.random.RandomState(123)
        # self.remove_outlier = remove_outlier
        # self.valid_obj_idxs = valid_obj_idxs
        # self.augment = augment
        # self.crop = crop
        # camera_split = 'data' if camera == 'realsense' else 'data_kinect'
        # self.collision_labels = {}

        # f = open(os.path.join(CFGPATH, '{}_data_list.txt'.format(split)))
        # self.data_list = []

        # for x in tqdm(f.readlines(), desc = 'Loading data path and collision labels...'):
        #     for img_num in range(256):
        #         self.data_list.append([int(x.strip().split('_')[1]), img_num])
        # f.close()
        
        self.data_list = []
        if split == 'train':
            for scene_idx in range(0, 100):
                if scene_idx == 51:
                    continue
                for anno_idx in range(0, 256):
                    self.data_list.append([scene_idx, anno_idx])
        elif split == 'test_seen':
            for scene_idx in range(100, 130):
                for anno_idx in range(0, 256):
                    self.data_list.append([scene_idx, anno_idx])
        elif split == 'test_similiar':
            for scene_idx in range(130, 160):
                for anno_idx in range(0, 256):
                    self.data_list.append([scene_idx, anno_idx])
        elif split == 'test_novel':
            for scene_idx in range(160, 190):
                for anno_idx in range(0, 256):
                    self.data_list.append([scene_idx, anno_idx])
        else:
            raise NotImplementedError

        if split == 'train':
            random.shuffle(self.data_list)

    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        scene_idx, anno_idx = self.data_list[index][0], self.data_list[index][1]

        dump_dir = os.path.join(self.label_root, 'center_anno', 'scene_%d'%scene_idx, self.camera, '0255.npz')
        bbox_dir = os.path.join(self.label_root, 'bbox_anno', 'scene_%d'%scene_idx, self.camera, '0255.npz')
        center_dump = np.load(dump_dir)['arr_0'][anno_idx]
        bbox_dump = np.load(bbox_dir)['arr_0'][anno_idx]

        color_dir = os.path.join(self.data_root, 'scenes', 'scene_%04d'%scene_idx, self.camera, 'rgb', str(anno_idx).zfill(4)+'.png')
        depth_dir = os.path.join(self.data_root, 'scenes', 'scene_%04d'%scene_idx, self.camera, 'depth', str(anno_idx).zfill(4)+'.png')
        score_dir = os.path.join(self.label_root, 'score_maps', 'scene_%d'%scene_idx, self.camera, 'numpy', '%04d.npz'%anno_idx)
        
        # center_dir = os.path.join(self.label_root, 'center_anno', 'scene_%d'%scene_idx, self.camera, 'numpy', '%04d.npz'%anno_idx)
        # colli_dir = os.path.join(self.label_root, 'scene_%d'%scene_idx, self.camera, 'colli_mask', '%04d.npz'%anno_idx)
        # wrench_dir = os.path.join(self.label_root, 'scene_%d'%scene_idx, self.camera, 'wrench', '%04d.npz'%anno_idx)
        # mask_dir = os.path.join(self.label_root, 'scene_%d'%scene_idx, self.camera, 'label_mask', '%04d.npz'%anno_idx)

        # print('color_dir:', color_dir)
        # tic = time.time()
        color = cv2.imread(color_dir).astype(np.float32) / 255.0
        depth = cv2.imread(depth_dir, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        # toc = time.time()
        # print('input read time:', toc-tic)

        # tic = time.time()
        score = np.load(score_dir)['arr_0']
        # toc = time.time()
        # print('score map load time:', toc-tic)
        
        center_map = np.zeros_like(depth)
        for i in range(center_dump.shape[0]):
            center = center_dump[i]
            coord = [center[1], center[0]]
            if not self.adapt_radius:
                drawGaussian(center_map, coord, 1, 5)
            else:
                bbox = bbox_dump[i]
                bbox_h, bbox_w = bbox[2]-bbox[0], bbox[3]-bbox[1]
                radius = gaussian_radius((math.ceil(bbox_h), math.ceil(bbox_w)))
                draw_msra_gaussian(center_map, coord, radius)
        

        if self.split == 'train':
            color, depth, score, center_map = self.crop_array(color, depth, score, center_map, self.dim)
            # color_noise = np.random.normal(scale=0.03, size=color.shape).astype(np.float32)
            depth_noise = np.random.normal(scale=0.03, size=depth.shape).astype(np.float32)

            # color = color + color_noise
            color = color_aug(self.data_rng, color)
            depth = depth + depth_noise

        return color, depth, score, center_map, (scene_idx, anno_idx)
    

    def debug(self, saveroot):
        
        for index in range(30):
            scene_idx, anno_idx = self.data_list[index][0], self.data_list[index][1]

            dump_dir = os.path.join(self.label_root, 'center_anno', 'scene_%d'%scene_idx, self.camera, '0255.npz')
            bbox_dir = os.path.join(self.label_root, 'bbox_anno', 'scene_%d'%scene_idx, self.camera, '0255.npz')
            center_dump = np.load(dump_dir)['arr_0'][anno_idx]
            bbox_dump = np.load(bbox_dir)['arr_0'][anno_idx]

            color_dir = os.path.join(self.data_root, 'scenes', 'scene_%04d'%scene_idx, self.camera, 'rgb', str(anno_idx).zfill(4)+'.png')
            depth_dir = os.path.join(self.data_root, 'scenes', 'scene_%04d'%scene_idx, self.camera, 'depth', str(anno_idx).zfill(4)+'.png')
            score_dir = os.path.join(self.label_root, 'score_maps', 'scene_%d'%scene_idx, self.camera, 'numpy', '%04d.npz'%anno_idx)
            
            # center_dir = os.path.join(self.label_root, 'center_anno', 'scene_%d'%scene_idx, self.camera, 'numpy', '%04d.npz'%anno_idx)
            # colli_dir = os.path.join(self.label_root, 'scene_%d'%scene_idx, self.camera, 'colli_mask', '%04d.npz'%anno_idx)
            # wrench_dir = os.path.join(self.label_root, 'scene_%d'%scene_idx, self.camera, 'wrench', '%04d.npz'%anno_idx)
            # mask_dir = os.path.join(self.label_root, 'scene_%d'%scene_idx, self.camera, 'label_mask', '%04d.npz'%anno_idx)

            # print('color_dir:', color_dir)
            color = cv2.imread(color_dir).astype(np.float32)
            depth = cv2.imread(depth_dir, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
            score = np.load(score_dir)['arr_0']

            center_map = np.zeros_like(depth)
            for i in range(center_dump.shape[0]):
                center = center_dump[i]
                coord = [center[1], center[0]]
                if not self.adapt_radius:
                    drawGaussian(center_map, coord, 1, 5)
                else:
                    bbox = bbox_dump[i]
                    bbox_h, bbox_w = bbox[2]-bbox[0], bbox[3]-bbox[1]
                    radius = gaussian_radius((math.ceil(bbox_h), math.ceil(bbox_w)))
                    draw_msra_gaussian(center_map, coord, radius)
        
            # center_map = center_map[..., 0]

            # color = cv2.resize(color, self.dim, interpolation=cv2.INTER_LINEAR)
            # depth = cv2.resize(depth, self.dim, interpolation=cv2.INTER_LINEAR)
            # score = cv2.resize(score, self.dim, interpolation=cv2.INTER_LINEAR)
            # center_map = cv2.resize(center_map, self.dim, interpolation=cv2.INTER_LINEAR)
            # wrench = cv2.resize(wrench, self.dim, interpolation=cv2.INTER_NEAREST)
            # # print('colli:', colli.dtype)
            # # print('colli:', type(colli))
            # colli = cv2.resize(colli.astype(np.int32), self.dim, interpolation=cv2.INTER_NEAREST)
            # mask = cv2.resize(mask.astype(np.int32), self.dim, interpolation=cv2.INTER_NEAREST)

            if self.split == 'train':
                color, depth, score, center_map = self.crop_array(color, depth, score, center_map, self.dim)
            # print('score1:', score.dtype)
            # score = score * colli
            # score = score.astype(np.float32)

            # idx0, idx1 = np.nonzero(mask)
            # print('idx0:', idx0)
            # print('idx1:', idx1)

            # print('idx0:', type(idx0))
            score_image = score
            wrench_image = center_map
            
            score_image *= 255
            score_image = score_image.clip(0, 255)
            score_image = score_image.astype(np.uint8)
            score_image = cv2.applyColorMap(score_image[..., np.newaxis], cv2.COLORMAP_RAINBOW)
            # rgb_image = score_image
            # score_image = np.array(score_image).astype(np.float32)
            rgb_image = 0.5 * color + 0.5 * score_image
            rgb_image = rgb_image.astype(np.uint8)
            im = Image.fromarray(rgb_image)
            
            visu_dir = os.path.join(saveroot, 'scene_'+str(scene_idx))
            os.makedirs(visu_dir, exist_ok=True)
            print('Saving:', visu_dir+'/score_%04d'%anno_idx+'.png')
            # start_time = time.time()
            im.save(visu_dir+'/score_%04d'%anno_idx+'.png')

            wrench_image *= 255
            
            wrench_image = wrench_image.clip(0, 255)
            wrench_image = wrench_image.astype(np.uint8)
            wrench_image = cv2.applyColorMap(wrench_image[..., np.newaxis], cv2.COLORMAP_RAINBOW)
            # rgb_image = wrench_image
            # wrench_image = np.array(wrench_image).astype(np.float32)
            rgb_image = 0.5 * color + 0.5 * wrench_image
            rgb_image = rgb_image.astype(np.uint8)
            im = Image.fromarray(rgb_image)
            
            visu_dir = os.path.join(saveroot, 'scene_'+str(scene_idx))
            os.makedirs(visu_dir, exist_ok=True)
            print('Saving:', visu_dir+'/wrench_%04d'%anno_idx+'.png')
            # start_time = time.time()
            im.save(visu_dir+'/wrench_%04d'%anno_idx+'.png')


    def augment(self, img, depth, score, center_map):
        input_h, input_w = img.shape[0], img.shape[1]
        s = max(input_h, input_w) * 1.0
        c = np.array([input_w / 2., input_h / 2.], dtype=np.float32)
        
        # flipped = False
        flip = 0.3

        s = s * np.random.choice(np.arange(0.8, 1.2, 0.1))
        w_border = self._get_border(128, img.shape[1])
        h_border = self._get_border(128, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
        
        if np.random.random() < flip:
            # flipped = True
            img = img[:, ::-1, :]
            depth = depth[:, ::-1]
            score = score[:, ::-1]
            center_map = center_map[:, ::-1]
            c[0] = input_w - c[0] - 1
        
        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        img = cv2.warpAffine(img, trans_input, 
                            (input_w, input_h),
                            flags=cv2.INTER_LINEAR)
        depth = cv2.warpAffine(depth, trans_input, 
                            (input_w, input_h),
                            flags=cv2.INTER_LINEAR)
        score = cv2.warpAffine(score, trans_input, 
                            (input_w, input_h),
                            flags=cv2.INTER_NEAREST)
        center_map = cv2.warpAffine(center_map, trans_input, 
                            (input_w, input_h),
                            flags=cv2.INTER_NEAREST)

        return img, depth, score, center_map

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i
    
    def crop_array(self, color, depth, score, center_map, t_size=(480, 480)):
        height, width = color.shape[0], color.shape[1]
        center_x = np.random.randint(t_size[1] // 2, width - t_size[1] // 2)
        center_y = np.random.randint(t_size[0] // 2, height - t_size[0] // 2)

        cropped_color = color[center_y-t_size[0]//2:center_y+t_size[0]//2, center_x-t_size[1]//2:center_x+t_size[1]//2]
        cropped_depth = depth[center_y-t_size[0]//2:center_y+t_size[0]//2, center_x-t_size[1]//2:center_x+t_size[1]//2]
        cropped_score = score[center_y-t_size[0]//2:center_y+t_size[0]//2, center_x-t_size[1]//2:center_x+t_size[1]//2]
        cropped_center_map = center_map[center_y-t_size[0]//2:center_y+t_size[0]//2, center_x-t_size[1]//2:center_x+t_size[1]//2]
        return cropped_color, cropped_depth, cropped_score, cropped_center_map

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3, ))
    image += np.dot(eigvec, eigval * alpha)
    return image

def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2
    return image1

def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    return blend_(alpha, image, gs[:, :, None])

def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha
    return image

def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    return blend_(alpha, image, gs_mean)

def color_aug(data_rng, image):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        image = f(data_rng, image, gs, gs_mean, 0.4)
    # image = lighting_(data_rng, image, 0.1, eig_val, eig_vec)
    return image

def color_aug2(data_rng, image, eig_val, eig_vec):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        image = f(data_rng, image, gs, gs_mean, 0.4)
    image = lighting_(data_rng, image, 0.1, eig_val, eig_vec)
    return image


if __name__ == "__main__":
    img_path = r'G:\DataSet\COCO\001.jpg'
    img = cv2.imread(img_path)

    print(img.max())
