import os
import numpy as np
from PIL import Image
import open3d as o3d
from utils.xmlhandler import xmlReader
import scipy.io as scio
from PIL import Image
import cv2
from transforms3d.euler import euler2mat
from multiprocessing import Process
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default='/DATA2/Benchmark/graspnet', help='Directory to save dataset')
parser.add_argument('--saveroot', default='/DATA2/Benchmark/suction/center_bbox_test', help='Directory to save bbox results')
parser.add_argument('--save_visu', action='store_true', help='Whether to save visualizations')
parser.add_argument('--camera', default='realsense', help='camera to use [default: realsense]')
parser.add_argument('--pool_size', type=int, default=10, help='How many threads to use')
FLAGS = parser.parse_args()


DATASET_ROOT = FLAGS.data_root
scenedir = FLAGS.data_root + '/scenes/scene_{}/{}'
bbox_saveroot = os.path.join(FLAGS.saveroot, 'bbox_anno')
center_saveroot = os.path.join(FLAGS.saveroot, 'center_anno')
visu_saveroot = os.path.join(FLAGS.saveroot, 'visu')
# mask_saveroot = ''


class CameraInfo():
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale


def transform_points(points, trans):
    ones = np.ones([points.shape[0],1], dtype=points.dtype)
    points_ = np.concatenate([points, ones], axis=-1)
    points_ = np.matmul(trans, points_.T).T
    return points_[:,:3]

def transform_normals(points, trans):
    ones = np.ones([points.shape[0],1], dtype=points.dtype)
    points_ = np.concatenate([points, ones], axis=-1)
    trans[:3, 3] = 0
    points_ = np.matmul(trans, points_.T).T
    return points_[:,:3]

def parse_posevector(posevector):
    mat = np.zeros([4,4],dtype=np.float32)
    alpha, beta, gamma = posevector[4:7]
    alpha = alpha / 180.0 * np.pi
    beta = beta / 180.0 * np.pi
    gamma = gamma / 180.0 * np.pi
    mat[:3,:3] = euler2mat(alpha, beta, gamma)
    mat[:3,3] = posevector[1:4]
    mat[3,3] = 1
    # obj_idx = id_scene2obj(int(posevector[0]))
    obj_idx = int(posevector[0])
    return obj_idx, mat


def generate_scene_model(dataset_root, scene_name, anno_idx, return_poses=False, camera='realsense'):

    # if align:
    #     camera_poses = np.load(os.path.join(dataset_root, 'scenes', scene_name, camera, 'camera_poses.npy'))
    #     camera_pose_origin = camera_poses[anno_idx]
    #     align_mat = np.load(os.path.join(dataset_root, 'scenes', scene_name, camera, 'cam0_wrt_table.npy'))
    #     camera_pose = np.matmul(align_mat,camera_pose_origin)

    scene_reader = xmlReader(os.path.join(dataset_root, 'scenes', scene_name, camera, 'annotations', '%04d.xml'%anno_idx))
    posevectors = scene_reader.getposevectorlist()
    obj_list = []
    mat_list = []
    model_list = []
    pose_list = []
    for posevector in posevectors:
        obj_idx, pose = parse_posevector(posevector)
        obj_list.append(obj_idx)
        mat_list.append(pose)

    for obj_idx, pose in zip(obj_list, mat_list):
        plyfile = os.path.join(dataset_root, 'models', '%03d'%obj_idx, 'nontextured.ply')
        model = o3d.io.read_point_cloud(plyfile)
        points = np.array(model.points)
        # if align:
        #     pose_align = np.dot(np.linalg.inv(camera_pose_origin), camera_pose)
        #     pose = np.dot(pose_align, pose)
        points = transform_points(points, pose)
        model.points = o3d.utility.Vector3dVector(points)
        # print('finish reading model')
        model_list.append(model)
        pose_list.append(pose)

    if return_poses:
        return model_list, obj_list, pose_list
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
    # camera_split = 'data' if camera == 'realsense' else 'data_kinect'
    meta_path = os.path.join(scenedir.format('%04d'%scene_idx, camera), 'meta', '%04d.mat'%(anno_idx))
    meta = scio.loadmat(meta_path)
    
    try:
        # obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
        # poses = meta['poses']
        intrinsics = meta['intrinsic_matrix']
        # factor_depth = meta['factor_depth']
    except Exception as e:
        print(repr(e))
        print(scene_idx)

    # intrinsics = np.load(os.path.join('camera_poses','{}_camK.npy'.format(camera)))
    # array([[631.54864502,   0.        , 638.43517329],
    #    [  0.        , 631.20751953, 366.49904066],
    #    [  0.        ,   0.        ,   1.        ]])
    # print('intrinsics:', intrinsics)
    fx, fy = intrinsics[0,0], intrinsics[1,1]
    cx, cy = intrinsics[0,2], intrinsics[1,2]
    s = 1000.0
    depth = s * points[:, 2] # point_z
    ###################################
    # x and y should be inverted here #
    ###################################
    
    x = points[:, 0] / points[:, 2] * fx + cx
    y = points[:, 1] / points[:, 2] * fy + cy
    # return x,y,depth
    return x.astype(np.int32), y.astype(np.int32), depth.astype(np.int32)


def get_model_grasps(datapath):
    dump = np.load(datapath)
    points = dump['points']
    normals = dump['normals']
    scores = dump['scores']
    collision = dump['collision']
    return points, normals, scores, collision


def get_center_bbox(scene_idx, camera='realsense'):
    print('Scene {}, {}'.format('scene_%04d'%scene_idx, camera))
    
    bbox_list_scene = []
    center_list_scene = []
    mask_list_scene = []
    
    for anno_idx in range(256):
        # camera_pose = camera_poses[anno_idx]
        # # print('camera pose')
        # # print(camera_pose)
        # if align:
        #     align_mat = np.load(os.path.join(DATASET_ROOT, 'scenes', 'scene_%04d'%scene_idx, camera, 'cam0_wrt_table.npy'))
        #     camera_pose = align_mat.dot(camera_pose)
        
        rgb_dir = os.path.join(scenedir.format('%04d'%scene_idx, camera), 'rgb', '%04d'%anno_idx+'.png')
        rgb_image = np.array(Image.open(rgb_dir), dtype=np.float32)

        model_list, _, _ = generate_scene_model(DATASET_ROOT, 'scene_%04d'%scene_idx, anno_idx, return_poses=True, camera=camera)

        bbox_list_single = []
        center_list_single = []
        mask_list_single = []

        for i, model in enumerate(model_list):
            
            points = np.array(model.points)
            # print('points:', points.shape)
            center = np.mean(points, keepdims=True, axis=0)

            x, y, _ = points2depth(points, scene_idx, camera)
            # print('center:', center.shape)
            center_x, center_y, _ = points2depth(center, scene_idx, camera)

            valid_y = y
            valid_x = x

            min_x = valid_x.min()
            min_y = valid_y.min()
            
            max_x = valid_x.max()
            max_y = valid_y.max()

            assert center_x[0] > min_x and center_x[0] < max_x, 'center x out of bbox'
            assert center_y[0] > min_y and center_y[0] < max_y, 'center y out of bbox'

            if not ((center_y[0] >= 0 ) & (center_y[0] < 720) & (center_x[0] >= 0 ) & (center_x[0] < 1280)):
                mask_list_single.append(0)
            else:
                mask_list_single.append(1)

            bbox = np.array([min_y, min_x, max_y, max_x], dtype=np.int32)
            bbox_list_single.append(bbox[np.newaxis, :])

            center_pix = np.concatenate([center_y, center_x], axis=0)[np.newaxis, :]
            # print('center_pix:', center_pix.shape)
            center_list_single.append(center_pix)
            if scene_idx < 10 and FLAGS.save_visu:
                rgb_image[max(min_y, 0): min(max_y, 720), max(min_x, 0): min(max_x, 1280), :] *= 0.5
                cv2.circle(rgb_image, (center_x, center_y), 10, (255,0,0), -1)

        bbox_single = np.concatenate(bbox_list_single, axis=0)[np.newaxis, :, :]                
        mask_single = np.array(mask_list_single, dtype=bool)[np.newaxis, :]
        bbox_list_scene.append(bbox_single)
        mask_list_scene.append(mask_single)
        # print('concatenate:', np.concatenate(center_list_single, axis=0).shape)
        center_single = np.concatenate(center_list_single, axis=0)[np.newaxis, :, :]
        center_list_scene.append(center_single)

        if anno_idx < 10 and FLAGS.save_visu:
            if (mask_single == 0).sum() == 0:
                rgb_image = rgb_image.astype(np.uint8)
                im = Image.fromarray(rgb_image)
                visu_dir = os.path.join(visu_saveroot, 'scene_'+str(scene_idx), camera)
                os.makedirs(visu_dir, exist_ok=True)
                print('Saving:', visu_dir+'/%04d'%anno_idx+'.png')
                im.save(visu_dir+'/%04d'%anno_idx+'.png')

    bbox_scene = np.concatenate(bbox_list_scene, axis=0)
    center_scene = np.concatenate(center_list_scene, axis=0)
    
    bbox_dir = os.path.join(bbox_saveroot, 'scene_'+str(scene_idx), camera)
    os.makedirs(bbox_dir, exist_ok=True)
    print('Saving:', bbox_dir + '/%04d'%anno_idx+'.npz')
    np.savez(bbox_dir + '/%04d'%anno_idx+'.npz', bbox_scene)

    center_dir = os.path.join(center_saveroot, 'scene_'+str(scene_idx), camera)
    os.makedirs(center_dir, exist_ok=True)
    print('Saving:', center_dir + '/%04d'%anno_idx+'.npz')
    np.savez(center_dir + '/%04d'%anno_idx+'.npz', center_scene)

    # mask_dir = os.path.join(mask_saveroot, 'scene_'+str(scene_idx), camera)
    # os.makedirs(mask_dir, exist_ok=True)
    # print('Saving:', mask_dir + '/%04d'%anno_idx+'.npz')
    # np.savez(mask_dir + '/%04d'%anno_idx+'.npz', mask_scene)


if __name__ == "__main__":
    
    camera = FLAGS.camera  

    scene_list = []
    for i in range(0, 100):
        scene_list.append(i)

    pool_size = FLAGS.pool_size
    pool_size = min(pool_size, len(scene_list))
    pool = []
    for _ in range(pool_size):
        scene_idx = scene_list.pop(0)
        # save_dir = '/data/fred/graspnet/scene_mask2/scene_{}'.format(scene_idx)
        pool.append(Process(target=get_center_bbox, args=(scene_idx,camera)))
    [p.start() for p in pool]
    while len(scene_list) > 0:
        for idx, p in enumerate(pool):
            if not p.is_alive():
                pool.pop(idx)
                scene_idx = scene_list.pop(0)
                # save_dir = '/data/fred/graspnet/scene_mask2/scene_{}'.format(scene_idx)
                p = Process(target=get_center_bbox, args=(scene_idx,camera))
                p.start()
                pool.append(p)
                break
    while len(pool) > 0:
        for idx, p in enumerate(pool):
            if not p.is_alive():
                pool.pop(idx)
                break
    
    # for scene_idx in scene_list:
    #     get_center_bbox(scene_idx, camera)

