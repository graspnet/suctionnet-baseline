import cv2
# import pcl
import time
import numpy as np
import open3d as o3d
from scipy.ndimage import generic_filter


def create_point_cloud_from_depth_image(depth, camera, organized=True):
    assert(depth.shape[0] == camera.height and depth.shape[1] == camera.width)
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud

def stdFilt(img, wlen):
    '''
    cal std filter of img
    :param img:
    :param wlen:  kernal size
    :return:
    '''
    wmean, wsqrmean = (cv2.boxFilter(x, -1, (wlen, wlen), borderType=cv2.BORDER_REFLECT) for x in (img, img*img))
    # pdb.set_trace()
    return np.sqrt(abs(wsqrmean - wmean*wmean))

def estimate_suction(depth_img, obj_mask, camera_info):
    point_cloud = create_point_cloud_from_depth_image(depth_img, camera_info)
    # print('point_cloud:', point_cloud.shape)

    # valid_idx = obj_mask & (point_cloud[..., 2] != 0)
    valid_idx = np.zeros_like(obj_mask, dtype=np.bool)
    coord1, coord2 = np.nonzero(obj_mask)
    coord1_min, coord1_max = coord1.min(), coord1.max()
    coord2_min, coord2_max = coord2.min(), coord2.max()
    valid_idx[coord1_min:coord1_max+1, coord2_min:coord2_max+1] = 1
    valid_idx = valid_idx & (point_cloud[..., 2] != 0)
    # print(point_cloud[obj_mask].shape)
    height, width, _ = point_cloud.shape

    # point_cloud = point_cloud.reshape(-1, 3)
    point_cloud_valid = point_cloud[valid_idx]
    # print('point_cloud_valid:', point_cloud_valid.shape)
    pc_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud_valid))
    # print('here')
    pc_o3d.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(224), fast_normal_computation=False)
    pc_o3d.orient_normals_to_align_with_direction(np.array([0., 0., -1.]))
    pc_o3d.normalize_normals()
    normals = np.array(pc_o3d.normals).astype(np.float32)
    
    # p = pcl.PointCloud(point_cloud_valid.astype(np.float32))
    # norm = p.make_NormalEstimation()
    # norm.set_KSearch(224)
    # normals = norm.compute()
    # normals = normals.to_array()
    # normals = normals[:, 0:3]
    # normals[normals[..., -1] > 0] = -normals[normals[..., -1] > 0]
    # normals = normals / np.linalg.norm(normals, axis=-1)[:, np.newaxis]

    # point_cloud = point_cloud.reshape(height, width, 3)
    # valid_idx = obj_mask & (point_cloud[..., 2] != 0)
    # normals = normals.reshape(height, width, 3)

    normal_map = np.zeros([height, width, 3], dtype=np.float32)
    normal_map[valid_idx] = normals
    # print('filter start')
    # tic = time.time()
    # mean_normal_std = np.mean(generic_filter(normal_map, np.std, size=25), axis=2)
    mean_normal_std = np.mean(stdFilt(normal_map, 25), axis=2)
    # toc = time.time()
    # print('filter time:', toc - tic)
    heatmap = 1 - mean_normal_std / np.max(mean_normal_std)
    heatmap[~valid_idx] = 0

    return heatmap, normal_map, point_cloud



