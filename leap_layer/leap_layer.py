# leap_hand layer for torch
import torch
import trimesh
import os
import numpy as np
import copy
import pytorch_kinematics as pk


import sys
sys.path.append('./')
from layer_asset_utils import save_part_mesh, sample_points_on_mesh, sample_visible_points

BASE_DIR = os.path.split(os.path.abspath(__file__))[0]
# All lengths are in mm and rotations in radians


class LeapHandLayer(torch.nn.Module):
    def __init__(self, to_mano_frame=True, show_mesh=False, device='cuda'):
        super().__init__()

        self.show_mesh = show_mesh
        self.to_mano_frame = to_mano_frame
        self.device = device

        urdf_path = os.path.join(BASE_DIR, '../assets/leap_hand.urdf')
        self.chain = pk.build_chain_from_urdf(open(urdf_path).read()).to(device=device)

        self.joints_lower = self.chain.low
        self.joints_upper = self.chain.high
        self.joints_mean = (self.joints_lower + self.joints_upper) / 2
        self.joints_range = self.joints_mean - self.joints_lower
        self.joint_names = self.chain.get_joint_parameter_names()
        self.n_dofs = self.chain.n_joints  # only used here for robot hand with no mimic joint

        # print(self.chain.get_links())
        self.link_dict = {}
        for link in self.chain.get_links():
            self.link_dict[link.name] = link.visuals[0].geom_param[0].split('/')[-1]

        # order in palm -> thumb -> index -> middle -> ring [-> pinky(little)]
        self.order_keys = [
            'palm_lower',  # palm
            'pip_4', 'thumb_pip', 'thumb_dip', 'thumb_fingertip',  # thumb
            'mcp_joint', 'pip', 'dip', 'fingertip',  # index
            'mcp_joint_2', 'pip_2', 'dip_2', 'fingertip_2',  # middle
            'mcp_joint_3', 'pip_3', 'dip_3', 'fingertip_3',  # ring
        ]

        # transformation for align the robot hand to mano hand frame, used for
        self.to_mano_transform = torch.eye(4).to(torch.float32).to(device)
        if self.to_mano_frame:
            self.to_mano_transform[:3, :] = torch.tensor([[-1, 0, 0, 0],
                                                           [0, 0, 1, 0.0175],
                                                           [0, 1, 0, 0.0375]])

        self.register_buffer('base_2_world', self.to_mano_transform)

        if not (os.path.exists(os.path.abspath(os.path.dirname(__file__)) + '/../assets/hand_meshes_cvx')
                and os.path.exists(os.path.abspath(os.path.dirname(__file__)) + '/../assets/hand_points')
                and os.path.exists(os.path.abspath(os.path.dirname(__file__)) + '/../assets/hand_composite_points')
                and os.path.exists(os.path.abspath(os.path.dirname(__file__)) + '/../assets/visible_point_indices')
                and os.path.exists(os.path.abspath(os.path.dirname(__file__)) + '/../assets/hand.obj')
                and os.path.exists(os.path.abspath(os.path.dirname(__file__)) + '/../assets/hand_all_zero.obj')
        ):
            # for first time run to generate contact points on the hand, set the self.make_contact_points=True
            self.make_contact_points = True
            self.create_assets()
        else:
            self.make_contact_points = False

        self.meshes = self.load_meshes()
        self.hand_segment_indices = self.get_hand_segment_indices()

    def create_assets(self):
        '''
        To create needed assets for the first running.
        Should run before first use.
        '''
        self.to_mano_transform = torch.eye(4).to(torch.float32).to(device)
        pose = torch.from_numpy(np.identity(4)).to(device).reshape(-1, 4, 4).float()
        theta = np.zeros((1, self.n_dofs), dtype=np.float32)

        save_part_mesh()
        sample_points_on_mesh()

        show_mesh = self.show_mesh
        self.show_mesh = True
        self.make_contact_points = True

        self.meshes = self.load_meshes()
        mesh = self.get_forward_hand_mesh(pose, theta)[0]
        parts = mesh.split()

        new_mesh = trimesh.boolean.boolean_manifold(parts, 'union')
        new_mesh.export(os.path.join(BASE_DIR, '../assets/hand.obj'))

        self.show_mesh = True
        self.make_contact_points = False
        self.meshes = self.load_meshes()
        mesh = self.get_forward_hand_mesh(pose, theta)[0]
        mesh.export(os.path.join(BASE_DIR, '../assets/hand_all_zero.obj'))

        self.show_mesh = False
        self.make_contact_points = True
        self.meshes = self.load_meshes()

        self.get_forward_vertices(pose, theta)      # SAMPLE hand_composite_points
        sample_visible_points()

        self.show_mesh = True
        self.make_contact_points = False

        self.to_mano_transform[:3, :] = torch.tensor([[-1, 0, 0, 0],
                                                      [0, 0, 1, 0.0175],
                                                      [0, 1, 0, 0.0375]])
        self.meshes = self.load_meshes()
        mesh = self.get_forward_hand_mesh(pose, theta)[0]
        mesh.export(os.path.join(BASE_DIR, '../assets/hand_to_mano_frame.obj'))

        self.make_contact_points = False
        self.show_mesh = show_mesh

    def load_meshes(self):
        mesh_dir = os.path.dirname(os.path.realpath(__file__)) + "/../assets/hand_meshes/"
        meshes = {}
        for key, value in self.link_dict.items():
            mesh_filepath = os.path.join(mesh_dir, value)
            link_pre_transform = self.chain.find_link(key).visuals[0].offset
            if self.show_mesh:
                mesh = trimesh.load(mesh_filepath)
                if self.make_contact_points:
                    mesh = trimesh.load(mesh_filepath.replace('assets/hand_meshes/', 'assets/hand_meshes_cvx/'))

                #     # mesh = mesh.convex_hull
                #     mesh_ = mesh.convex_decomposition(
                #         maxConvexHulls=16 if key == 'base_link' else 2,
                #         resolution=800000 if key == 'base_link' else 1000,
                #         minimumVolumePercentErrorAllowed=0.1 if key == 'base_link' else 10,
                #         maxRecursionDepth=10 if key == 'base_link' else 4,
                #         shrinkWrap=True, fillMode='flood', maxNumVerticesPerCH=32,
                #         asyncACD=True, minEdgeLength=2, findBestPlane=False
                #     )
                #     mesh = np.sum(mesh_)
                verts = link_pre_transform.transform_points(torch.FloatTensor(np.array(mesh.vertices)))


                temp = torch.ones(mesh.vertices.shape[0], 1).float()
                vertex_normals = link_pre_transform.transform_normals(torch.FloatTensor(copy.deepcopy(mesh.vertex_normals)))
                meshes[key] = [
                    torch.cat((verts, temp), dim=-1).to(self.device),
                    mesh.faces,
                    torch.cat((vertex_normals, temp), dim=-1).to(self.device).to(torch.float)
                ]
            else:
                vertex_path = mesh_filepath.replace('hand_meshes', 'hand_points').replace('.stl', '.npy').replace('.STL', '.npy')
                assert os.path.exists(vertex_path)
                points_info = np.load(vertex_path)

                link_pre_transform = self.chain.find_link(key).visuals[0].offset
                if self.make_contact_points:
                    idxs = np.arange(len(points_info))
                else:
                    idxs = np.load(os.path.dirname(os.path.realpath(__file__)) + '/../assets/visible_point_indices/{}.npy'.format(key))

                verts = link_pre_transform.transform_points(torch.FloatTensor(points_info[idxs, :3]))
                # print(key, value, verts.shape)
                vertex_normals = link_pre_transform.transform_normals(torch.FloatTensor(points_info[idxs, 3:6]))

                temp = torch.ones(idxs.shape[0], 1)

                meshes[key] = [
                    torch.cat((verts, temp), dim=-1).to(self.device),
                    torch.zeros([0]),  # no real meaning, just for placeholder
                    torch.cat((vertex_normals, temp), dim=-1).to(torch.float).to(self.device)
                ]

        return meshes

    def get_hand_segment_indices(self):
        hand_segment_indices = {}
        start = torch.tensor(0, dtype=torch.long, device=self.device)
        for link_name in self.order_keys:
            end = torch.tensor(self.meshes[link_name][0].shape[0], dtype=torch.long, device=self.device) + start
            hand_segment_indices[link_name] = [start, end]
            start = end.clone()
        return hand_segment_indices

    def forward(self, theta):
        """
        Args:
            theta (Tensor (batch_size x 15)): The degrees of freedom of the Robot hand.
       """
        ret = self.chain.forward_kinematics(theta)
        return ret

    def get_hand_mesh(self, pose, ret):
        bs = pose.shape[0]

        meshes = []
        for key in self.order_keys:
            rotmat = ret[key].get_matrix()
            # rotmat = torch.matmul(pose, torch.matmul(self.to_mano_transform.T, rotmat))
            rotmat = torch.matmul(pose, torch.matmul(self.to_mano_transform, rotmat))

            vertices = self.meshes[key][0]
            batch_vertices = torch.matmul(rotmat, vertices.transpose(0, 1)).transpose(1, 2)[..., :3]
            face = self.meshes[key][1]
            sub_meshes = [trimesh.Trimesh(vertices.cpu().numpy(), face) for vertices in batch_vertices]

            meshes.append(sub_meshes)

        hand_meshes = []
        for j in range(bs):
            hand = [meshes[i][j] for i in range(len(meshes))]
            hand_mesh = np.sum(hand)
            hand_meshes.append(hand_mesh)
        return hand_meshes

    def get_forward_hand_mesh(self, pose, theta):
        outputs = self.forward(theta)

        hand_meshes = self.get_hand_mesh(pose, outputs)

        return hand_meshes

    def get_forward_vertices(self, pose, theta):
        outputs = self.forward(theta)

        verts = []
        verts_normal = []

        # for key, item in self.meshes.items():
        for key in self.order_keys:
            rotmat = outputs[key].get_matrix()
            rotmat = torch.matmul(pose, torch.matmul(self.to_mano_transform, rotmat))

            vertices = self.meshes[key][0]
            vertex_normals = self.meshes[key][2]
            batch_vertices = torch.matmul(rotmat, vertices.transpose(0, 1)).transpose(1, 2)[..., :3]
            verts.append(batch_vertices)

            if self.make_contact_points:
                if not os.path.exists('../assets/hand_composite_points'):
                    os.makedirs('../assets/hand_composite_points', exist_ok=True)
                np.save('../assets/hand_composite_points/{}.npy'.format(key),
                        batch_vertices.squeeze().cpu().numpy())
            rotmat[:, :3, 3] = 0
            batch_vertex_normals = torch.matmul(rotmat, vertex_normals.transpose(0, 1)).transpose(1, 2)[..., :3]
            verts_normal.append(batch_vertex_normals)

        verts = torch.cat(verts, dim=1).contiguous()
        verts_normal = torch.cat(verts_normal, dim=1).contiguous()
        return verts, verts_normal


class LeapAnchor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # vert_idx
        vert_idx = np.array([
            # thumb finger
            1382, 1522, 1541, 1667, 1493,
            428, 179,

            # index finger
            1806, 2289, 2408, 2405, 2442,  # 2324
            19,

            # middle finger
            2504, 3016, 3164, 3049, 3060,
            364, 626,

            # ring finger
            3454, 3756, 3863, 3844, 3915,
            0, 0,  # place holder

            # little finger
            0, 0, 0, 0, 0,  # place holder

            # # plus
            2420, 2332, 2131, 2241,  # 2440  2463
            3129, 3133, 2895, 3005,
            3815, 3778, 3644, 3713,
            0, 0,  # place holder

        ])
        # vert_idx = np.load(os.path.join(BASE_DIR, 'anchor_idx.npy'))
        self.register_buffer("vert_idx", torch.from_numpy(vert_idx).long())

    def forward(self, vertices):
        """
        vertices: TENSOR[N_BATCH, 4040, 3]
        """
        anchor_pos = vertices[:, self.vert_idx, :]
        return anchor_pos

    def pick_points(self, vertices: np.ndarray):
        import open3d as o3d
        print("")
        print(
            "1) Please pick at least three correspondences using [shift + left click]"
        )
        print("   Press [shift + right click] to undo point picking")
        print("2) Afther picking points, press q for close the window")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # user picks points
        vis.destroy_window()
        print(vis.get_picked_points())
        return vis.get_picked_points()


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    show_mesh = False
    to_mano_frame = True
    allegro = LeapHandLayer(show_mesh=show_mesh, to_mano_frame=to_mano_frame, device=device)

    pose = torch.from_numpy(np.identity(4)).to(device).reshape(-1, 4, 4).float()
    theta = np.zeros((1, 16), dtype=np.float32)
    theta[0, :4] = np.array([0.0, -0.0, 0, 0])
    theta = torch.from_numpy(theta).to(device)

    # mesh version
    if show_mesh:
        mesh = allegro.get_forward_hand_mesh(pose, theta)[0]
        mesh.show()
    else:

        verts, normals = allegro.get_forward_vertices(pose, theta)
        pc = trimesh.PointCloud(verts.squeeze().cpu().numpy(), colors=(0, 255, 255))

        ray_visualize = trimesh.load_path(np.hstack((verts[0].detach().cpu().numpy(),
                                                     verts[0].detach().cpu().numpy() + normals[0].detach().cpu().numpy() * 0.01)).reshape(-1, 2, 3))
        scene = trimesh.Scene([pc, ray_visualize])
        scene.show()

        mesh = trimesh.load(os.path.join(BASE_DIR, '../assets/hand_to_mano_frame.obj'))

        anchor_layer = LeapAnchor()
        anchors = anchor_layer(verts).squeeze().cpu().numpy()
        pc_anchors = trimesh.PointCloud(anchors, colors=(0, 0, 255))
        ray_visualize = trimesh.load_path(np.hstack((verts[0].detach().cpu().numpy(),
                                                     verts[0].detach().cpu().numpy() + normals[
                                                         0].detach().cpu().numpy() * 0.01)).reshape(-1, 2, 3))

        scene = trimesh.Scene([mesh, pc, pc_anchors, ray_visualize])
        scene.show()

