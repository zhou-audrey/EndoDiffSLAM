import pymeshfix as mf
import pyvista as pv
import numpy as np
import trimesh
import open3d as o3d
import subprocess
import sys
import os
import shutil


def extract_holes(mesh_root, timestamp):
    prefix = f'{int(timestamp):05d}'
    mesh_file = mesh_root + f'{prefix}_mesh.ply'
    mesh = pv.read(mesh_file)

    meshfix = mf.MeshFix(mesh)
    holes = meshfix.extract_holes()

    p = pv.Plotter()
    p.add_mesh(mesh, color=True)
    p.add_mesh(holes, color="r", line_width=8)
    p.enable_eye_dome_lighting()
    tube_radius = 0.01
    tube = holes.tube(radius=tube_radius)
    tube.point_data["RGB"] = np.full((tube.n_points, 3), [255, 0, 0], dtype=np.uint8)
    tube.save(mesh_root + "extract_holes.ply")

    mesh1 = trimesh.load(mesh_file)
    mesh2 = trimesh.load(mesh_root + 'extract_holes.ply')
    merged_mesh = trimesh.util.concatenate([mesh1, mesh2])
    merged_mesh.export(mesh_root + "path_to_merged_model.ply")


class PointPickingVisualizer:
    def __init__(self, point_cloud):
        self.point_cloud = point_cloud
        self.picked_indices = []

    def run(self):
        picked_indices = self.pick_points()
        assert len(picked_indices) >= 4 or len(picked_indices) == 0
        if picked_indices:
            for idx in picked_indices:
                self.picked_indices.append(idx)
                point = np.asarray(self.point_cloud.points)[idx]
            if len(self.picked_indices) >= 4:
                print("4 points selected. Exiting...")
        return self.picked_indices

    def pick_points(self):
        print("Press at least select 4 points by [shift+left]. Cancel by [shift+right]. Close by [Q]")
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(self.point_cloud)
        vis.run()
        vis.destroy_window()
        return vis.get_picked_points()


def calculate_extrinsic(mesh_root, timestamp, diffusion_idx):
    extract_holes(mesh_root, timestamp)
    point_cloud = o3d.io.read_point_cloud(mesh_root + "path_to_merged_model.ply")

    picker = PointPickingVisualizer(point_cloud)
    picked_indices = picker.run()
    if picked_indices:
        points = np.asarray(point_cloud.points)
        selected_coords = points[picked_indices]
        print(f"Selected point indices: {picked_indices}")
        print(f"Selected point coordinates:\n{selected_coords}")
        mean_coords = np.mean(selected_coords, axis=0)
        print(f"mean_coords:\n{mean_coords}")
    else:
        print("No points selected.")
        return None

    mesh = o3d.io.read_triangle_mesh(mesh_root + "path_to_merged_model.ply")
    mesh.compute_vertex_normals()
    vertex_normals = np.asarray(mesh.vertex_normals)
    indices_to_output = picked_indices
    for idx in indices_to_output:
        print(f"Vertex {idx} normal: {vertex_normals[idx]}")
    selected_normals = vertex_normals[indices_to_output]
    mean_normal = np.mean(selected_normals, axis=0)
    print(f"Mean normal of selected vertices: {mean_normal}")

    mean_coords = np.array(mean_coords)
    mean_normal = np.array(mean_normal)
    distance = 0.3
    new_point = mean_coords + mean_normal * distance
    print("New point coordinates:", new_point)

    t = np.array(new_point)
    n = np.array(mean_normal) * (-1)
    n_normalized = n / np.linalg.norm(n)
    y_world = np.array([0, 1, 0])
    x_camera = np.cross(y_world, n_normalized)
    x_camera = x_camera / np.linalg.norm(x_camera)
    y_camera = np.cross(n_normalized, x_camera)
    R = np.column_stack((x_camera, y_camera, n_normalized))
    extrinsic_matrix = np.column_stack((R, t))
    print("Extrinsic Matrix:")
    print(extrinsic_matrix)

    extrinsic_matrix = np.array(extrinsic_matrix)
    pose_file = f"{diffusion_idx}.npy"
    pose_path = os.path.join("diffusion_views", pose_file)
    np.save(pose_path, extrinsic_matrix)
    print("Extrinsic matrix saved as", pose_path)

    return extrinsic_matrix


def run_diffusion(diffusion_idx, diffusion_num):
    print("run diffusion")
    diffusion_num1 = diffusion_num
    files = os.listdir("diffusion_views")
    for file in files:
        filename, extension = os.path.splitext(file)
        new_filename = f"{int(filename):03d}{extension}"
        source_path = os.path.join("diffusion_views", file)
        target_path = os.path.join("Endo-Escher/demo/GSO30/endo/render_mvs_25/model", new_filename)
        shutil.copy(source_path, target_path)
        if filename in diffusion_idx:
            if not os.path.exists("diffusion_views_fix"):
                os.makedirs("diffusion_views_fix")
            new_filename = f"{diffusion_num1}{extension}"
            target_path_fix = os.path.join("diffusion_views_fix", new_filename)
            shutil.copy(source_path, target_path_fix)
            diffusion_num1 += 1

    conda_path = subprocess.check_output(["which", "conda"]).decode().strip()
    command = f"{conda_path} run -n eschernet bash eval_eschernet.sh"
    new_working_directory = "Endo-Escher/"
    result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=new_working_directory)
    if result.returncode == 0:
        print("Command executed successfully!")
        print("Output:")
        print(result.stdout)
    else:
        print("Command execution failed!")
        print("Error message:")
        print(result.stderr)
    print(result.stdout)

    files = os.listdir("Endo-Escher/logs_6DoF/GSO25/N5M25/endo")
    for file in files:
        source_path = os.path.join("Endo-Escher/logs_6DoF/GSO25/N5M25/endo", file)
        target_path = os.path.join("diffusion_views", file)
        shutil.copy(source_path, target_path)
        filename, extension = os.path.splitext(file)
        if filename in diffusion_idx:
            new_filename = f"{diffusion_num}{extension}"
            target_path_fix = os.path.join("diffusion_views_fix", new_filename)
            shutil.copy(source_path, target_path_fix)
            diffusion_num += 1

    return diffusion_num