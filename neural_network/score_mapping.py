import os
import numpy as np
from PIL import Image
from utils.xmlhandler import xmlReader
import scipy.io as scio
from PIL import Image
import cv2
from transforms3d.euler import euler2mat, quat2mat
from multiprocessing import Process
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default='', help='Directory to save dataset')
parser.add_argument('--saveroot', default='', help='Directory to save score map results')
parser.add_argument('--save_visu', action='store_true', help='Whether to save visualizations')
parser.add_argument('--camera', default='realsense', help='Camera to use [default: realsense]')
parser.add_argument('--sigma', type=int, default=4, help='Gaussian Kernel sigma')
parser.add_argument('--pool_size', type=int, default=10, help='How many threads to use')
FLAGS = parser.parse_args()


DATASET_ROOT = FLAGS.data_root
scenedir = FLAGS.data_root + '/scenes/scene_{}/{}'
labeldir = os.path.join(DATASET_ROOT, 'seal_label')
saveroot = os.path.join(FLAGS.saveroot, 'score_maps')
colli_root = os.path.join(DATASET_ROOT, 'suction_collision_label')

class CameraInfo():
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale

def parse_posevector(posevector):
    mat = np.zeros([4,4],dtype=np.float32)
    alpha, beta, gamma = posevector[4:7]
    alpha = alpha / 180.0 * np.pi
    beta = beta / 180.0 * np.pi
    gamma = gamma / 180.0 * np.pi
    mat[:3,:3] = euler2mat(alpha, beta, gamma)
    mat[:3,3] = posevector[1:4]
    mat[3,3] = 1
    obj_idx = int(posevector[0])
    return obj_idx, mat

def transform_points(points, trans):
    ones = np.ones([points.shape[0],1], dtype=points.dtype)
    points_ = np.concatenate([points, ones], axis=-1)
    points_ = np.matmul(trans, points_.T).T
    return points_[:,:3]


def generate_scene_model(dataset_root, scene_name, anno_idx, return_poses=False, 
                            align=False, camera='realsense'):

    print('Scene {}, {}'.format(scene_name, camera))
    scene_reader = xmlReader(os.path.join(dataset_root, 'scenes', scene_name, camera, 'annotations', '%04d.xml'%anno_idx))
    posevectors = scene_reader.getposevectorlist()
    obj_list = []
    mat_list = []
    model_list = []
    # pose_list = []
    for posevector in posevectors:
        obj_idx, pose = parse_posevector(posevector)
        obj_list.append(obj_idx)
        mat_list.append(pose)

    if return_poses:
        return model_list, obj_list, mat_list # pose_list
    else:
        return model_list


def create_point_cloud_from_depth_image(depth, camera, organized=True):
    assert(depth.shape[0] == camera.height and depth.shape[1] == camera.width)
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / camera.scale
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud


def points2depth(points,scene_idx, camera='kinect', anno_idx=0):
    meta_path = os.path.join(scenedir.format('%04d'%scene_idx, camera), 'meta', '%04d.mat'%(anno_idx))
    meta = scio.loadmat(meta_path)
    
    try:
        intrinsics = meta['intrinsic_matrix']
    except Exception as e:
        print(repr(e))
        print(scene_idx)

    fx, fy = intrinsics[0,0], intrinsics[1,1]
    cx, cy = intrinsics[0,2], intrinsics[1,2]
    s = 1000.0
    depth = s * points[:, 2] # point_z
    x = points[:, 0] / points[:, 2] * fx + cx
    y = points[:, 1] / points[:, 2] * fy + cy
    return x.astype(np.int32), y.astype(np.int32), depth.astype(np.int32)


def get_model_grasps(datapath):
    dump = np.load(datapath)
    points = dump['points']
    normals = dump['normals']
    scores = dump['scores']
    collision = dump['collision']
    return points, normals, scores, collision

def gaussian_kernel(kernel_size, sigma):
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    
    sum_val = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i-center, j-center
            
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            sum_val += kernel[i, j]
    
    kernel = kernel / sum_val
    return kernel.astype(np.float32)

def uniform_kernel(kernel_size):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    kernel = kernel / kernel_size**2

    return kernel

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
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)) * score
    g = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    tmp_img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g
    img += tmp_img

    return img

def score_mapping(scene_idx, camera):
    if not os.path.exists(colli_root+'/{:04d}_collision.npz'.format(scene_idx)):
        print('Missing ' + colli_root+'/{:04d}_collision.npz'.format(scene_idx))
        return
        
    collision_dump = np.load(colli_root+'/{:04d}_collision.npz'.format(scene_idx))

    for anno_idx in range(256):
        
        rgb_dir = os.path.join(scenedir.format('%04d'%scene_idx, camera), 'rgb', '%04d'%anno_idx+'.png')
        rgb_image = np.array(Image.open(rgb_dir), dtype=np.float32)

        # k_size = 31
        # sigma = 5
        score_image = np.zeros([720, 1280], dtype=np.float32)

        _, obj_list, pose_list = generate_scene_model(DATASET_ROOT, 'scene_%04d'%scene_idx, anno_idx, return_poses=True, 
                                                        align=True, camera=camera) 
        
        for i in range(len(obj_list)):
            obj_idx = obj_list[i]
            trans = pose_list[i]
            sampled_points, normals, scores, _ = get_model_grasps('%s/%03d_seal.npz'%(labeldir, obj_idx))
            
            collision = collision_dump['arr_{}'.format(i)]

            sampled_points = transform_points(sampled_points, trans)
            
            assert sampled_points.shape[0] > 0, "No points"

            normals = transform_points(-normals, trans)
            normals = normals / np.linalg.norm(normals, axis=-1)[:, np.newaxis]

            vert_idx = normals[:, 2] > 0.5
            sampled_points = sampled_points[vert_idx]
            scores = scores[vert_idx]
            collision = collision[vert_idx]
            x, y, depth = points2depth(sampled_points, scene_idx, camera)
            # print(y)

            valid_idx = (y >= 0 ) & (y < 720) & (x >= 0 ) & (x < 1280)
            valid_y = y[valid_idx]
            valid_x = x[valid_idx]
            valid_depth = depth[valid_idx]
            if valid_depth.shape[0] == 0:
                continue
            
            valid_score = scores[valid_idx]
            valid_colli = collision[valid_idx]

            sort_idx = np.argsort(valid_depth)
            sort_y = valid_y[sort_idx]
            sort_x = valid_x[sort_idx]
            sort_score = valid_score[sort_idx]
            sort_colli = valid_colli[sort_idx]
            sort_coord = np.concatenate([sort_x[:, np.newaxis], sort_y[:, np.newaxis]], axis=-1)

            unique_coord, unique_idx = np.unique(sort_coord, return_index=True, axis=0)
            unique_score = sort_score[unique_idx]
            unique_score = sort_score[unique_idx] * (~sort_colli[unique_idx])

            for i in range(unique_coord.shape[0]):
                score = unique_score[i]
                coord = [unique_coord[i, 0], unique_coord[i, 1]]
                drawGaussian(score_image, coord, score, FLAGS.sigma)

        numpy_dir = os.path.join(saveroot, 'scene_'+str(scene_idx), camera, 'numpy')
        os.makedirs(numpy_dir, exist_ok=True)
        print('Saving:', numpy_dir + '/%04d'%anno_idx+'.npz')
        np.savez(numpy_dir + '/%04d'%anno_idx+'.npz', score_image)

        if anno_idx < 3:
            score_image *= 255
            
            score_image = score_image.clip(0, 255)
            score_image = score_image.astype(np.uint8)
            score_image = cv2.applyColorMap(score_image, cv2.COLORMAP_RAINBOW)
            rgb_image = 0.5 * rgb_image + 0.5 * score_image
            rgb_image = rgb_image.astype(np.uint8)
            im = Image.fromarray(rgb_image)
            
            visu_dir = os.path.join(saveroot, 'scene_'+str(scene_idx), camera, 'visu')
            os.makedirs(visu_dir, exist_ok=True)
            print('Saving:', visu_dir+'/%04d'%anno_idx+'.png')
            im.save(visu_dir+'/%04d'%anno_idx+'.png')


if __name__ == "__main__":
    
    align = True
    camera = FLAGS.camera   
    
    scene_list = []
    for i in range(0, 100):
        scene_list.append(i)

    pool_size = FLAGS.pool_size
    pool_size = min(pool_size, len(scene_list))
    pool = []
    for _ in range(pool_size):
        scene_idx = scene_list.pop(0)
        pool.append(Process(target=score_mapping, args=(scene_idx,camera)))
    [p.start() for p in pool]
    while len(scene_list) > 0:
        for idx, p in enumerate(pool):
            if not p.is_alive():
                pool.pop(idx)
                scene_idx = scene_list.pop(0)
                p = Process(target=score_mapping, args=(scene_idx,camera))
                p.start()
                pool.append(p)
                break
    while len(pool) > 0:
        for idx, p in enumerate(pool):
            if not p.is_alive():
                pool.pop(idx)
                break
    
