from mmgen.datasets.builder import DATASETS
from mmcv.parallel import DataContainer as DC
from .colmap_utils import *
from torch.utils.data import Dataset
import os
import json
import numpy as np
import torch
from torchvision import transforms as T
import tqdm
from PIL import Image
import cv2
from scipy.spatial.transform import Slerp, Rotation
from scipy.optimize import least_squares
# from .colmap_utils import *
def rotmat(a, b):
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	# handle exception for the opposite direction input
	if c < -1 + 1e-10:
		return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def center_poses(poses, pts3d=None, enable_cam_center=False):
    def normalize(v):
        return v / (np. linalg.norm(v) + 1e-10)
    if pts3d is None or enable_cam_center:
        center = poses[:, :3, 3].mean(0)
    else:
        center = pts3d.mean(0)
    
    up = normalize(poses[:, :3, 1].mean(0))
    R = rotmat(up, [0, 0, 1])
    R = np.pad(R, [0, 1])
    R[-1, -1] = 1

    poses[:, :3, 3] -= center
    poses_centered = R @ poses
    if pts3d is not None:
        pts3d_centered = (pts3d - center) @ R[:3, :3].T
        return poses_centered, pts3d_centered
    
    return poses_centered

def compute_target_center(poses):
    positions = poses[:, :3, 3]
    rotations = poses[:, :3, :3]
    z_dirs = -rotations @ np.array([0, 0, 1])

    def residuals(c):
        diff = c - positions
        cross = np.linalg.norm(np.cross(diff, z_dirs), axis=-1)
        return cross
    c_guess = np.mean(positions, axis=0)

    results = least_squares(residuals, c_guess)
    return results.x


def interpolate_poses(poses, num_dense=60):
    positions = poses[:, :3, 3]
    rotations = poses[:, :3, :3]

    # Generate new positions.

    # radii = np.linalg.norm(positions[:, :2], axis=-1)
    # avg_radius = np.mean(radii)
    # avg_height = np.mean(positions[:, 2])

    # angles = np.linspace(0, 2*np.pi, num_dense, endpoint=False)
    # new_postions = np.column_stack([
    #     avg_radius * np.cos(angles),
    #     avg_radius * np.sin(angles),
    #     np.full(num_dense, avg_height)
    # ]
    # )

    # get the new cameara pose
    if not np.allclose(rotations[0], rotations[-1]):
        rotations = np.vstack([rotations, rotations[0][None]])
        positions = np.vstack([positions, positions[0][None]])
    
    rots = Rotation.from_matrix(rotations)
    slerp = Slerp(np.linspace(0, 1, len(rots)), rots)
    new_rotations = slerp(np.linspace(0, 1, num_dense, endpoint=False))

    radii = np.linalg.norm(positions[:, :2], axis=-1)
    avg_radius = np.mean(radii)
    avg_height = np.mean(positions[:, 2])

    angles = np.linspace(0, 2*np.pi, num_dense, endpoint=False)
    new_positions = np.column_stack([
        avg_radius * np.cos(angles),
        avg_radius * np.sin(angles),
        np.full(num_dense, avg_height)
    ])

    new_poses = np.tile(np.eye(4), (num_dense, 1, 1))
    new_poses[:, :3, 3] = new_positions
    poses_temp = new_rotations.as_matrix()
    poses_temp = poses_temp[:, :, [2 ,1, 0]]
    poses_temp[:, :, 2] *= -1
    new_poses[:, :3, :3] = poses_temp

    # Refine the pose.
    z_interp = poses_temp @ np.array([0, 0, 1])
    target_center = compute_target_center(new_poses)
    # target_center = positions.mean(0)
    z_target = target_center - new_poses[:, :3, 3]
    z_target = z_target / np.linalg.norm(z_target, axis=-1,keepdims=True)
    
    blender_factor = 1
    z_final = (1 - blender_factor) * z_interp + blender_factor * z_target
    z_final = z_final / np.linalg.norm(z_final, axis=-1, keepdims=True)
    
    for i in range(num_dense):
        z_axis = z_final[i] 
        x_axis = -np.cross(z_axis, new_poses[i, :3, 1])
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        new_poses[i, :3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    
    new_poses[:, :3, 0] *= -1
    new_poses[:, :3, 2] *= -1
    return new_poses


def selected_upper_down_positions(poses):
    z_axis = poses[:, 2, 2]
    # avg_z_axis = np.mean(z_axis)
    avg_z_axis = torch.mean(z_axis)
    print(avg_z_axis)
    upper_mask = z_axis > avg_z_axis
    down_mask = z_axis < avg_z_axis
    upper_poses = poses[upper_mask]
    down_poses = poses[down_mask]
    print(f'upper poses: {upper_poses.shape}, down poses: {down_poses.shape}')
    poses_upper_interpolate = interpolate_poses(upper_poses, num_dense=5)
    poses_down_interpolate = interpolate_poses(down_poses, num_dense=30)
    return poses_upper_interpolate, poses_down_interpolate


@DATASETS.register_module()
class CustomDataset(Dataset):
    def __init__(self, root_dir: str, split: str , world_scale: float=1.0, rgba:bool =True, n_views=6):
        super().__init__()
        self.root_dir = root_dir
        self.world_scale = world_scale
        self.rgba = rgba
        self.split = split
        self.downscale = 1
        self.n_views = n_views
        self.dust3r_path = os.path.join(self.root_dir)

        if self.dust3r_path is None:
            raise ValueError(f'Cannot find dust3r path under {self.dust3r_path}')
        with open(os.path.join(self.dust3r_path, f'dust3r_{self.n_views}.json'), 'r') as f:
            self.dust3r_data = json.load(f)
        self.H = int(round(self.dust3r_data[0]['height'] / self.downscale))
        self.W = int(round(self.dust3r_data[0]['width'] / self.downscale))
        print(f'[INFO] CustomDataset: image H = {self.H}, image W = {self.W}')

        # load image folder 
        img_names = [names['img_name'] for names in self.dust3r_data]
        img_folder = os.path.join(self.root_dir, 'images')
        img_paths = np.array([os.path.join(img_folder, name) for name in img_names])
        
        # load mask folder
        mask_folder = os.path.join(self.root_dir, 'masks')
        if os.path.exists(mask_folder):
            print(f'[INFO] Use mask under folder {mask_folder}')
            mask_paths = np.array([os.path.join(mask_folder, os.path.splitext(name)[0] + '.png') for name in img_names])
        # load intrinsics
        intrinsics = []
        for k in self.dust3r_data:
            # dust3r do not offer the cx, cy so use the colmap estimation for use.
            cx = self.W / 2
            cy = self.H / 2
            fx = 1001.2491989135742
            fy = 1001.2491989135742

            intrinsics.append(np.array([fx, fy, cx, cy], dtype=np.float32))
        self.intrinsics = torch.from_numpy(np.stack(intrinsics))

        # read poses
        poses = []
        for item in self.dust3r_data:
            pose = np.eye(4)
            position = np.array(item['position'])
            rotation = np.array(item['rotation'])
            pose[:3, 3] = position.T
            pose[:3, :3] = rotation
            poses.append(pose)

        # Do global transformation for poses
        poses = np.stack(poses, axis=0)
        self.poses = center_poses(poses)
        print(f'[INFO] CustomDataSet: poses shape = {self.poses.shape}')

        # Rectify the conventions.
        self.poses[:, :3, 1:3] *= -1
        self.poses = self.poses[:, [1, 0, 2, 3], :]
        self.poses[:, 2] *= -1
        
        if self.world_scale == 1:
            self.scale = 1 / np.linalg.norm(self.poses[:, :3, 3], axis=-1).min()
        self.poses[:, :3, 3] *= self.scale
        
        # make split
        # for the dust3r do not make split for now...
        if self.split == 'train':
            self.poses = self.poses
            self.intrinsics = self.intrinsics
            img_paths = img_paths
            if mask_paths is not None:
                mask_paths = mask_paths
        elif self.split == 'val':
            self.poses = self.poses
            self.intrinsics = self.intrinsics
            img_paths = img_paths
            if mask_paths is not None:
                mask_paths = mask_paths
        
        # read images
        self.define_transforms()
        self.images = []
        for i, f in enumerate(tqdm.tqdm(img_paths, desc=f'Loading {self.split} data')):
            image = Image.open(f)
            image = self.transform(image)
            image = image.permute(1, 2, 0)

            if mask_paths is not None:
                m_path = mask_paths[i]
                mask = cv2.imread(m_path, cv2.IMREAD_UNCHANGED)
                mask = (mask / 255).astype(np.float32)

                if len(mask.shape) == 2:
                    mask = mask[..., None]
                # blend A to RGB
                mask = torch.from_numpy(mask)
                image = image * mask + (1 - mask)

                if rgba:
                    image = torch.cat([image, mask[..., :1]], dim=-1)
            
            self.images.append(image)
        
        self.poses = torch.from_numpy(self.poses.astype(np.float32))

        if self.images is not None:
            self.images = torch.stack(self.images, dim=0)
        
        # _*, self.interpolate_dense_pose = interpolate_poses(self.poses)
        self.upper_poses, self.interpolate_dense_poses = selected_upper_down_positions(self.poses)
        # print(self.interpolate_dense_pose.shape)

    def define_transforms(self):
        self.transform = T.ToTensor()
    
    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        return dict(
            scene_id=DC(index, cpu_only=True),
            scene_name=DC(self.root_dir, cpu_only=True),
            cond_imgs=np.array(self.images, np.float32),
            cond_poses=np.array(self.poses, np.float32),
            cond_intrinsics=np.array(self.intrinsics, np.float32)
        )


@DATASETS.register_module()
class ColmapDataset(Dataset):
    def __init__(self, root_dir: str, split: str, world_scale: float = 1.0, rgba: bool = True):
        super().__init__()
        self.root_dir = root_dir
        self.world_scale = world_scale
        self.rgba = rgba
        self.split = split
        self.downscale = 1
        # load camera poses
        self.colmap_path = os.path.join(self.root_dir, "colmap_sparse", "0")
        if self.colmap_path is None:
            raise ValueError(f"Cannot find colmap path under {self.colmap_path}")
        
        camdata = read_cameras_binary(os.path.join(self.colmap_path, "cameras.bin"))
        
        # load image size

        self.H = int(round(camdata[1].height / self.downscale))
        self.W = int(round(camdata[1].width / self.downscale))
        print(f'[INFO] ColmapDataset: image H = {self.H}, image W = {self.W}')

        #read image paths
        imdata = read_images_binary(os.path.join(self.colmap_path, "images.bin"))
        imkeys = np.array(sorted(imdata.keys()))

        img_names = [os.path.basename(imdata[k].name) for k in imkeys]
        img_folder = os.path.join(self.root_dir, "images")
        img_paths = np.array([os.path.join(img_folder, name) for name in img_names])
        #keep the existing imgs
        exist_mask = np.array([os.path.exists(f) for f in img_paths])
        print(f'[INFO] {exist_mask.sum()} images exists in all {exist_mask.shape[0]} colmap entries')
        imkeys = imkeys[exist_mask]
        img_paths = img_paths[exist_mask]

        # load mask folder
        mask_folder = os.path.join(self.root_dir, 'mask')
        if os.path.exists(mask_folder):
            print(f'[INFO] Use mask under folder {mask_folder}')
            mask_paths = np.array([os.path.join(self.root_dir, 'mask', os.path.splitext(os.path.basename(p))[0] + '.png') for p in img_paths])
        else:
            mask_paths = None
        
        # read intrinsics
        intrinsics = []
        for k in imkeys:
            cam = camdata[imdata[k].camera_id]
            if cam.model in ['SIMPLE_RADIAL', 'SIMPLE_PINHOLE']:
                fl_x = fl_y = cam.params[0] / self.downscale
                cx = cam.params[1] / self.downscale
                cy = cam.params[2] / self.downscale
            elif cam.model in ['PINHOLE', 'OPENCV']:
                fl_x = cam.params[0] / self.downscale
                fl_y = cam.params[1] / self.downscale
                cx = cam.params[2] / self.downscale
                cy = cam.params[3] / self.downscale
            else:
                raise ValueError(f"Unsupported colmap camera model: {cam.model}")
            intrinsics.append(np.array([fl_x, fl_y, cx, cy], dtype=np.float32))
        self.intrinsics = torch.from_numpy(np.stack(intrinsics))

        # read poses
        poses = []
        for k in imkeys:
            P = np.eye(4, dtype=np.float64)
            P[:3, :3] = imdata[k].qvec2rotmat()
            P[:3, 3] = imdata[k].tvec
            poses.append(P)
        poses = np.linalg.inv(np.stack(poses, axis=0))

        # Do global transformation for poses:
        ptsdata = read_points3d_binary(os.path.join(self.colmap_path, 'points3D.bin'))
        ptskeys = np.array(sorted(ptsdata.keys()))
        pts3d = np.array([ptsdata[k].xyz for k in ptskeys])
        self.ptserr = np.array([ptsdata[k].error for k in ptskeys])
        self.mean_ptserr = np.mean(self.ptserr)

        # first center pose
        self.poses, self.pts3d = center_poses(poses, pts3d, enable_cam_center=True)
        print(f'[INFO] ColmapDataset: load poses {self.poses.shape}, points {self.pts3d.shape}')
        # Do not rectify the conventions.
        self.poses[:, :3, 1:3] *= -1
        self.poses = self.poses[:, [1, 0, 2, 3], :]
        self.poses[:, 2] *= -1

        self.pts3d = self.pts3d[:, [1, 0, 2]]
        self.pts3d[:, 2] *= -1

        # Rectify convention(not used and should be vertified)
        if self.world_scale == 1:
            # avg_position = np.linalg.norm(self.poses[:, :3, 3], axis=-1)

            # print(f'average pos is {avg_position}')
            self.scale = 1 / np.linalg.norm(self.poses[:, :3, 3], axis=-1).min()
            
            print(f'[INFO] ColmapDataset: auto-scale {self.scale:.4f}')
        self.poses[:, :3, 3] *= self.scale
        self.pts3d *= self.scale


        # make split
        all_ids = np.arange(len(img_paths))
        # This is the configuration for the duoji
        val_ids = np.array([10, 13])
        # This is the configuration for the motor_paper
        # val_ids = np.array([6, 18, 17, 1, 12, 5])
        # This is the configuration for the aniya
        # val_ids = np.array([20, 10, 6, 11, 9, 4])
        # This is the configuration for the model
        # val_ids = np.array([6, 3, 9, 5, 13, 1])

        if self.split == 'train':
            train_ids = np.array([i for i in all_ids if i not in val_ids])
            # This is the train idx for the duoji
            self.poses = self.poses[train_ids]
            self.intrinsics = self.intrinsics[train_ids]
            img_paths = img_paths[train_ids]
            if mask_paths is not None:
                mask_paths = mask_paths[train_ids]
            # Add depth info for train.
            self.dense_depth_info = []
            depth_folder = os.path.join(self.root_dir, 'depth')
            if os.path.exists(depth_folder):
                print(f'[INFO] use depth_Anything V2 under folder {depth_folder}')
                depth_paths = np.array([os.path.join(self.root_dir, 'depth', os.path.splitext(os.path.basename(p))[0] + '.png') for p in img_paths])
            else:
                depth_paths = None
            # load depth images.
            for i, f in enumerate(tqdm.tqdm(depth_paths, desc=f'Loading depth data')):
                depth_image = cv2.imread(f, cv2.IMREAD_UNCHANGED)
                depth_image = depth_image[:, : , :1] / 255 # depth_image = depth_image / 255 or not?
                self.dense_depth_info.append(depth_image)
            self.dense_depth_info = torch.from_numpy(np.stack(self.dense_depth_info, axis=0))
        elif self.split == 'val':
            self.poses = self.poses[val_ids]
            self.intrinsics = self.intrinsics[val_ids]
            img_paths = img_paths[val_ids]
            if mask_paths is not None:
                mask_paths = mask_paths[val_ids]
            self.dense_depth_info=None
        # read images
        self.define_transforms()
        self.images = []
        for i, f in enumerate(tqdm.tqdm(img_paths, desc=f'Loading {self.split} data')):
            # image = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            image = Image.open(f)
            image = self.transfrom(image) # [Channel=3, H, W]
            image = image.permute(1, 2, 0)
            if mask_paths is not None:
                m_path = mask_paths[i]
                mask = cv2.imread(m_path, cv2.IMREAD_UNCHANGED)
                mask = (mask / 255).astype(np.float32)
                if len(mask.shape) == 2:
                    mask = mask[..., None]
                #blend A to RGB
                mask = torch.from_numpy(mask)
                image = image * mask + (1 - mask)

                if rgba:
                    image = torch.cat([image, mask[..., :1]], dim=-1)
            
            self.images.append(image)

        self.poses = torch.from_numpy(self.poses.astype(np.float32))
        
        if self.images is not None:
            self.images = torch.stack(self.images, dim=0) #[N 1080 1920 4]
        

    def define_transforms(self):
        self.transfrom = T.ToTensor()

    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        return dict(
            scene_id=DC(index, cpu_only=True),
            scene_name=DC(self.root_dir, cpu_only=True),
            cond_imgs=np.array(self.images, np.float32),
            cond_depth=np.array(self.dense_depth_info, np.float32),
            cond_poses=np.array(self.poses, np.float32),
            cond_intrinsics=np.array(self.intrinsics, np.float32)
        )
