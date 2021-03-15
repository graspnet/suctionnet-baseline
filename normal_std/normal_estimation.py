import os
import cv2
import open3d as o3d
import numpy as np
import scipy.io as scio
from policy import estimate_suction


class CameraInfo():
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale

def create_point_cloud_from_depth_image(depth, camera, organized=True):
    assert(depth.shape[0] == camera.height and depth.shape[1] == camera.width)
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    # points_z = depth / camera.scale
    points_z = depth
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud

def grid_sample(pred_score_map, down_rate=20, topk=512):
    num_row = pred_score_map.shape[0] // down_rate
    num_col = pred_score_map.shape[1] // down_rate

    idx_list = []
    for i in range(num_row):
        for j in range(num_col):
            pred_score_grid = pred_score_map[i*down_rate:(i+1)*down_rate, j*down_rate:(j+1)*down_rate]
            # print('pred_score_grid:', pred_score_grid.shape)
            max_idx = np.argmax(pred_score_grid)
            
            max_idx = np.array([max_idx // down_rate, max_idx % down_rate]).astype(np.int32)
            
            max_idx[0] += i*down_rate
            max_idx[1] += j*down_rate
            # print('max_idx:', max_idx)
            idx_list.append(max_idx[np.newaxis, ...])
    
    idx = np.concatenate(idx_list, axis=0)
    # print('idx:', idx.shape)
    suction_scores = pred_score_map[idx[:, 0], idx[:, 1]]
    # print('suction_scores:', suction_scores.shape)
    sort_idx = np.argsort(suction_scores)
    sort_idx = sort_idx[::-1]

    sort_idx_topk = sort_idx[:topk]

    suction_scores_topk = suction_scores[sort_idx_topk]
    idx0_topk = idx[:, 0][sort_idx_topk]
    idx1_topk = idx[:, 1][sort_idx_topk]

    return suction_scores_topk, idx0_topk, idx1_topk

if __name__ == "__main__":
    dataset_root = r'G:\MyProject\data\Grasping\graspnet'
    camera = 'kinect'
    scene_idx = 100
    anno_idx = 0

    depth_file = os.path.join(dataset_root, 'scenes', 'scene_%04d'%scene_idx, camera, 'depth', '%04d.png'%anno_idx)
    meta_file = os.path.join(dataset_root, 'scenes', 'scene_%04d'%scene_idx, camera, 'meta', '%04d.mat'%anno_idx)
    segmask_file = os.path.join(dataset_root, 'scenes', 'scene_%04d'%scene_idx, camera, 'label', '%04d.png'%anno_idx)

    meta = scio.loadmat(meta_file)
    intrinsics = meta['intrinsic_matrix']
    fx, fy = intrinsics[0,0], intrinsics[1,1]
    cx, cy = intrinsics[0,2], intrinsics[1,2]
    width = 1280
    height = 720
    s = 1000.0
    camera_info = CameraInfo(width, height, fx, fy, cx, cy, s)

    depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    seg_mask = cv2.imread(segmask_file, cv2.IMREAD_UNCHANGED).astype(np.bool)
    # heatmap = np.ones_like(depth, dtype=np.float32)
    print('depth:', depth.max())

    depth = depth.clip(0, 1)
    point_cloud = create_point_cloud_from_depth_image(depth, camera_info)
    print('point cloud:', point_cloud.max())
    height, width, _ = point_cloud.shape
    heatmap, normals, point_cloud = estimate_suction(depth, seg_mask, camera_info)

    suction_scores, idx0, idx1 = grid_sample(heatmap, down_rate=10, topk=1024)
    print('idx0 min', idx0.min())
    print('idx0 max', idx0.max())
    # suction_scores = pred_score_map[idx0, idx1]
    # suction_directions = normals[idx0, idx1, :]
    suction_points = point_cloud[idx0, idx1, :]
    
    # toc_sub = time.time()
    # print('create point cloud time: ', toc_sub - tic_sub)

    # tic_sub = time.time()
    point_cloud = point_cloud.reshape(-1, 3)
    pc_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud))
    print('pc_o3d:', np.array(pc_o3d.points).astype(np.float32).shape)
    pc_sampled = pc_o3d.voxel_down_sample(0.003)
    pc_points = np.array(pc_sampled.points).astype(np.float32)
    pc_points = np.concatenate([suction_points, pc_points], axis=0)
    print('pc_points:', pc_points.shape)
    pc_sampled.points = o3d.utility.Vector3dVector(pc_points)
    pc_sampled.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(0.012), fast_normal_computation=False)
    pc_sampled.orient_normals_to_align_with_direction(np.array([0., 0., -1.]))
    pc_sampled.normalize_normals()
    pc_normals = np.array(pc_sampled.normals).astype(np.float32)
    suction_normals = pc_normals[:suction_points.shape[0], :]

    # # pc_o3d.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(224), fast_normal_computation=False)
    # pc_o3d.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(0.02), fast_normal_computation=False)
    # pc_o3d.orient_normals_to_align_with_direction(np.array([0., 0., -1.]))
    # pc_o3d.normalize_normals()
    # normals = np.array(pc_o3d.normals).astype(np.float32)
    # points = np.array(pc_o3d.points).astype(np.float32)

    normals = np.zeros([height, width, 3], dtype=np.float32)
    points = np.zeros([height, width, 3], dtype=np.float32)

    points[idx0, idx1, :] = suction_points
    normals[idx0, idx1, :] = suction_normals

    # print('normals:', normals.shape)
    print(points.max())

    normals = normals[seg_mask, :]
    points = points[seg_mask, :]
    # print('normals:', normals.shape)

    print('nonzero:', np.sum(normals[:, 2]!=0))
    normals = normals[normals[:, 2]!=0, :]
    points = points[points[:, 2]!=0, :]
    sampled_idx = np.random.choice(normals.shape[0], 50)
    normals_sampled = normals[sampled_idx, :]
    points_sampled = points[sampled_idx, :]

    # sampled_idx = np.random.choice(suction_normals.shape[0], 50)
    # normals_sampled = suction_normals[sampled_idx, :]
    # points_sampled = suction_points[sampled_idx, :]

    arrow_list = []
    for idx in range(len(points_sampled)):
        suction_point = points_sampled[idx]
        suction_normal = normals_sampled[idx]
        suction_normal = suction_normal / np.linalg.norm(suction_normal)
        # print(idx)
        # print('sction point:', suction_point)
        # print('suction normal:', suction_normal)

        arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.005, cone_radius=0.0075, 
                                                        cylinder_height=0.025, cone_height=0.02)
        arrow_points = np.asarray(arrow.vertices)

        new_z = suction_normal
        new_y = np.array((new_z[1], -new_z[0], 0), dtype=np.float64)
        new_y = new_y / np.linalg.norm(new_y)
        new_x = np.cross(new_y, new_z)

        R = np.c_[new_x, np.c_[new_y, new_z]]
        arrow_points = np.dot(R, arrow_points.T).T + suction_point[np.newaxis,:]
        arrow.vertices = o3d.utility.Vector3dVector(arrow_points)
        arrow_list.append(arrow)
    
    o3d.visualization.draw_geometries([pc_o3d, *arrow_list], width=1280, height=720)

