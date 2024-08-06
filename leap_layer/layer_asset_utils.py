import trimesh
import os
import numpy as np
from mesh_to_sdf import get_surface_point_cloud
from scipy.spatial import KDTree
import point_cloud_utils as pcu
import trimesh.sample
import coacd
import open3d as o3d


def o3d_vox_downsample(points, voxel_size=0.005):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    downsampled_pc = pc.voxel_down_sample(voxel_size)
    points = np.asarray(downsampled_pc.points)
    return points

def o3d_uniform_downsample(points, K=5):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    downsampled_pc = pc.uniform_down_sample(every_k_points=K)
    points = np.asarray(downsampled_pc.points)
    return points


def save_part_mesh(dst='../assets/hand_meshes', method='convexhull', threshold=0.3):
    for root, dirs, files in os.walk(dst):
        for filename in files:
            if filename.endswith('.obj') or filename.endswith('.stl'):
                filepath = os.path.join(root, filename)
                mesh = trimesh.load(filepath, force='mesh')
                if 'palm' in filepath:
                    tmp_method = 'convexhull'
                else:
                    tmp_method = method
                if tmp_method == 'coacd':
                    mesh = coacd.Mesh(mesh.vertices, mesh.faces)
                    parts = coacd.run_coacd(mesh, threshold=threshold, apx_mode='ch')
                    meshes = []
                    for part in parts:
                        mesh = trimesh.Trimesh(vertices=part[0], faces=part[1])
                        meshes.append(mesh)
                    new_mesh = trimesh.boolean.boolean_manifold(meshes, 'union')
                elif tmp_method =='convexhull':
                    new_mesh = mesh.convex_hull
                else:
                    raise ValueError('method must be coacd or convexhull')

                if not os.path.exists(dst.replace('hand_meshes', 'hand_meshes_cvx')):
                    os.makedirs(dst.replace('hand_meshes', 'hand_meshes_cvx'))
                new_filepath = filepath.replace('hand_meshes', 'hand_meshes_cvx').replace('.obj', '.stl')
                print('save to path:', new_filepath)
                new_mesh.export(new_filepath)

    print('save finished')


def save_part_convex_hull_mesh(dst='../assets/hand_meshes'):
    for root, dirs, files in os.walk(dst):
        for filename in files:
            if filename.endswith('.stl'):
                filepath = os.path.join(root, filename)
                mesh = trimesh.load_mesh(filepath)
                convex_mesh = mesh.convex_hull
                if not os.path.exists(dst.replace('hand_meshes', 'hand_meshes_cvx')):
                    os.makedirs(dst.replace('hand_meshes', 'hand_meshes_cvx'))
                new_filepath = filepath.replace('hand_meshes', 'hand_meshes_cvx')
                # print('save to path:', new_filepath)
                convex_mesh.export(new_filepath)
    print('save finished')


def sample_points_on_mesh(src_dir='../assets/hand_meshes_cvx'):
    for root, dirs, files in os.walk(src_dir):
        for filename in files:
            # print(filename)
            filepath = os.path.join(root, filename)
            mesh = trimesh.load_mesh(filepath)
            convex_mesh = mesh.copy()

            np.random.seed(0)
            points, idx = trimesh.sample.sample_surface_even(convex_mesh, 50000, radius=None)
            point_normals = convex_mesh.face_normals[idx]
            vis = False
            if vis:
                pc = trimesh.PointCloud(points, colors=(255, 255, 0))
                ray_visualization = trimesh.load_path(np.hstack((points,
                                                                points + point_normals / 100)).reshape(-1, 2, 3))
                scene = trimesh.Scene([pc, ray_visualization])
                scene.show()

            info = np.concatenate([points, point_normals], axis=-1)
            dst_dir = src_dir.replace('hand_meshes_cvx', 'hand_points')
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir, exist_ok=True)
            np.save(os.path.join(dst_dir, filename.replace('.stl', '.npy')), info)


def sample_visible_points(voxel_size=0.0055):
    count = 0

    hand = trimesh.load('../assets/hand.obj', force='mesh')
    result = get_surface_point_cloud(hand, scan_count=100, scan_resolution=400)
    points = np.array(result.points)
    points = pcu.downsample_point_cloud_on_voxel_grid(voxel_size/2, points)
    points = pcu.downsample_point_cloud_on_voxel_grid(voxel_size/2, points)

    dst = '../assets/hand_composite_points'

    for root, dirs, files in os.walk(dst):
        for filename in files:
            filepath = os.path.join(root, filename)
            point_info = np.load(filepath)
            point_tree = KDTree(data=points)
            dist, index = point_tree.query(point_info[:, :3], k=1)
            mask = dist < 0.0005
            index = index[mask]

            visible_points = point_info[mask]
            v_sampled = pcu.downsample_point_cloud_on_voxel_grid(voxel_size, visible_points[:, :3])
            v_sampled = pcu.downsample_point_cloud_on_voxel_grid(voxel_size, v_sampled)

            # v_sampled = o3d_vox_downsample(visible_points, 0.005)
            count += len(v_sampled)
            # pc = trimesh.PointCloud(v_sampled, colors=(255, 255, 0))
            # pc.show()

            _, v_sampled_idx = KDTree(point_info[:, :3]).query(v_sampled, k=1)
            # pc = trimesh.PointCloud(point_info[:, :3][v_sampled_idx])
            # pc.show()
            points[index] = np.array([-1e6, -1e6, 1e6])
            if not os.path.exists('../assets/visible_point_indices'):
                os.makedirs('../assets/visible_point_indices', exist_ok=True)
            np.save('../assets/visible_point_indices/{}.npy'.format(filename[:-4]), v_sampled_idx)
    print(count)


if __name__ == "__main__":
    # sample_points_on_mesh(src_dir='../assets/hand_meshes')
    # sample_points_on_mesh(src_dir='../assets/leap_hand_composite')
    # sample_visible_points()
    # save_part_convex_hull_mesh()
    # save_part_mesh()
    pass
