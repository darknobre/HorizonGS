import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, box
import math
import os

from utils.graphics_utils import BasicPointCloud

def extract_point_cloud_from_bound(pcd, x_bound, y_bound, plane_index):
    mask = (
        (pcd.points[:, plane_index[0]] >= x_bound[0])
        & (pcd.points[:, plane_index[0]] <= x_bound[1])
        & (pcd.points[:, plane_index[1]] >= y_bound[0])
        & (pcd.points[:, plane_index[1]] <= y_bound[1])
    )
    return BasicPointCloud(pcd.points[mask], pcd.colors[mask], pcd.normals[mask])
    
def extract_cams_from_bound(cams, x_bound, y_bound, plane_index):
    pose = np.array([camera.camera_center.cpu().numpy() for camera in cams])
    mask = (
        (pose[:, plane_index[0]] >= x_bound[0])
        & (pose[:, plane_index[0]] <= x_bound[1])
        & (pose[:, plane_index[1]] >= y_bound[0])
        & (pose[:, plane_index[1]] <= y_bound[1])
    )
    indices = np.where(mask)[0]
    return indices, [cams[idx] for idx in indices]

def extract_point_cloud(pcd, bbox):
    mask = (
        (pcd.points[:, 0] >= bbox[0])
        & (pcd.points[:, 0] <= bbox[1])
        & (pcd.points[:, 2] >= bbox[2])
        & (pcd.points[:, 2] <= bbox[3])
    )  # 筛选在范围内的点云，得到对应的mask
    points = pcd.points[mask]
    colors = pcd.colors[mask]
    normals = pcd.normals[mask]
    return points, colors, normals

def get_8_corner_points(pcd):
    x_list = pcd.points[:, 0]
    y_list = pcd.points[:, 1]
    z_list = pcd.points[:, 2]
    x_min, x_max, y_min, y_max, z_min, z_max = (
        min(x_list),
        max(x_list),
        min(y_list),
        max(y_list),
        min(z_list),
        max(z_list),
    )
    return {
        "minx_miny_minz": [x_min, y_min, z_min],
        "minx_miny_maxz": [x_min, y_min, z_max],
        "minx_maxy_minz": [x_min, y_max, z_min],
        "minx_maxy_maxz": [x_min, y_max, z_max],
        "maxx_miny_minz": [x_max, y_min, z_min],
        "maxx_miny_maxz": [x_max, y_min, z_max],
        "maxx_maxy_minz": [x_max, y_max, z_min],
        "maxx_maxy_maxz": [x_max, y_max, z_max],
    }
        
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def distance(a, b):
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def cross_product(a, b, c):
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)


def compare_angles(pivot, p1, p2):
    orientation = cross_product(pivot, p1, p2)
    if orientation == 0:
        return distance(pivot, p1) - distance(pivot, p2)
    return -1 if orientation > 0 else 1

def graham_scan(points):
    n = len(points)
    if n < 3:
        return "凸包需要至少3个点"

    pivot = min(points, key=lambda point: (point.y, point.x))
    points = sorted(points, key=lambda point: (np.arctan2(point.y - pivot.y, point.x - pivot.x), -point.y, point.x))

    stack = [points[0], points[1], points[2]]
    for i in range(3, n):
        while len(stack) > 1 and compare_angles(stack[-2], stack[-1], points[i]) > 0:
            stack.pop()
        stack.append(points[i])

    return stack


def plot_convex_hull(points, convex_hull, x, y):
    plt.figure()
    plt.scatter([p.x for p in points], [p.y for p in points], color="b", label="所有点")

    # 绘制凸包
    plt.plot(
        [p.x for p in convex_hull] + [convex_hull[0].x],
        [p.y for p in convex_hull] + [convex_hull[0].y],
        linestyle="-",
        color="g",
        label="篱笆边",
    )

    for i in range(len(convex_hull)):
        plt.plot(
            [convex_hull[i].x, convex_hull[(i + 1) % len(convex_hull)].x],
            [convex_hull[i].y, convex_hull[(i + 1) % len(convex_hull)].y],
            linestyle="-",
            color="g",
        )

    plt.plot(x, y)

    plt.show()


def run_graham_scan(points, W, H):
    """获取8个点围成的区域的凸包
    :param points 8个角点投影后的坐标
    :param W 图像宽度
    :param H 图像高度
    :return 凸包的点集 [x, y]
    """
    # points = [Point(point[0], point[1]) for point in points]
    # convex_hull = graham_scan(points)
    points = np.array(points)
    convex_hull = ConvexHull(np.array(points))
    # convex_hull_polygon = Polygon([(point[0], point[1]) for point in convex_hull])
    
    convex_hull_list = []
    # plt.plot(points[:, 0], points[:, 1], 'o')
    for i, j in zip(convex_hull.simplices, convex_hull.vertices):
        # plt.plot(points[i, 0], points[i, 1], 'k-')
        convex_hull_list.append(points[j])

    convex_hull_polygon = Polygon(convex_hull_list)
    image_bounds = box(0, 0, W, H)
    # x = [0, W, W, 0, 0]
    # y = [0, 0, H, H, 0]
    # plt.plot(x, y)
    # plt.show()
    # plot_convex_hull(points, convex_hull, x, y)
    # 计算凸包与图像边界的交集
    intersection = convex_hull_polygon.intersection(image_bounds)
    image_area = W * H  # 图像面积
    # 计算交集面积占图像面积的比例
    intersection_rate = intersection.area / image_area

    # print("intersection_area: ", intersection.area, " image_area: ", image_area, " intersection_rate: ", intersection_rate)
    return {
        "intersection_area": intersection.area,
        "image_area": image_area,
        "intersection_rate": intersection_rate,
    }

def point_in_image(camera, points):
    """使用投影矩阵将角点投影到二维平面"""
    # 获取点在图像平面的坐标
    R = camera.R
    T = camera.T
    w2c = np.eye(4)
    w2c[:3, :3] = np.transpose(R)
    w2c[:3, 3] = T

    # 这样写可能是不正确的，但在实验过程中没有发现明显的错误，不过还是进行修正
    # intrinsic_matrix = np.array([
    #    [fx, 0, camera.image_height // 2],
    #    [0, fy, camera.image_width // 2],
    #    [0, 0, 1]
    # ])

    # fix bug
    intrinsic_matrix = np.array([
        [camera.fx, 0, camera.cx], 
        [0, camera.fy, camera.cy], 
        [0, 0, 1]
    ]).astype(np.float32)

    points_camera = np.dot(w2c[:3, :3], points.T) + w2c[:3, 3:].reshape(3, 1)  # [3, n]
    points_camera = points_camera.T  # [n, 3]  [1, 3]
    points_camera = points_camera[np.where(points_camera[:, 2] > 0)]  # [n, 3]  这里需要根据z轴过滤一下点
    points_image = np.dot(intrinsic_matrix, points_camera.T)  # [3, n]
    points_image = points_image[:2, :] / points_image[2, :]  # [2, n]
    points_image = points_image.T  # [n, 2]

    mask = np.where(
        np.logical_and.reduce(
            (
                points_image[:, 0] >= 0,
                points_image[:, 0] < camera.image_height,
                points_image[:, 1] >= 0,
                points_image[:, 1] < camera.image_width,
            )
        )
    )[0]

    return points_image, points_image[mask], mask


def draw_partitions(partitions, png_name, labels, plane_index, logfolder):
    print(f"try to save {png_name}.png..")

    # Create a figure for all partitions
    fig, ax = plt.subplots()

    # Define a color map for different partitions
    colors = plt.cm.get_cmap("hsv", len(partitions))

    color_id = 1
    for partition_id, partition in partitions.items():
        x_coords = partition["pcd"].points[:, plane_index[0]]
        y_coords = partition["pcd"].points[:, plane_index[1]]

        # Scatter plot for points
        ax.scatter(x_coords, y_coords, c="blue", s=1)

        # Camera positions
        cam_x_coords = np.array([cam.camera_center[plane_index[0]].item() for cam in partition["cameras"]])
        cam_y_coords = np.array([cam.camera_center[plane_index[1]].item() for cam in partition["cameras"]])

        # Scatter plot for cameras
        ax.scatter(cam_x_coords, cam_y_coords, color="red", s=10, marker="x")

        # Draw a rectangle for the partition
        min_x, max_x = partition["bounds"][0][0], partition["bounds"][0][1]
        min_z, max_z = partition["bounds"][1][0], partition["bounds"][1][1]

        # Get a unique color for each partition
        rect_color = colors(int(color_id))
        rect = plt.Rectangle(
            (min_x, min_z), max_x - min_x, max_z - min_z, linewidth=1, edgecolor=rect_color, facecolor="none"
        )
        ax.add_patch(rect)

        # Add label inside the rectangle at the top
        ax.text((min_x + max_x) / 2, max_z + 0.5, f"{partition_id}", color=rect_color, fontsize=8, ha="center")
        color_id += 1

    ax.title.set_text("Plot of 2D Points and Cameras")
    ax.set_xlabel(labels[plane_index[0]])
    ax.set_ylabel(labels[plane_index[1]])
    fig.tight_layout()

    # Save the combined figure
    fig.savefig(os.path.join(logfolder, f"{png_name}.png"), dpi=200)
    plt.close(fig)

def draw_each_partition(partitions, png_name, plane_index, logfolder):
    print(f"try to save {png_name}.png..")
    save_path = os.path.join(logfolder, "partitions")
    os.makedirs(save_path, exist_ok=True)
    for partition_id, partition in partitions.items():
        x_coords = partition["pcd"].points[:, plane_index[0]]
        z_coords = partition["pcd"].points[:, plane_index[1]]
        fig, ax = plt.subplots()
        ax.scatter(x_coords, z_coords, c=(partition["pcd"].colors), s=1)
        ax.title.set_text("Plot of 2D Points")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Z-axis")
        fig.tight_layout()
        x_coords = np.array([cam.camera_center[plane_index[0]].item() for cam in partition["cameras"]])
        z_coords = np.array([cam.camera_center[plane_index[1]].item() for cam in partition["cameras"]])
        ax.scatter(x_coords, z_coords, color="red", s=1)

        # Draw a rectangle for the partition
        min_x, max_x = partition["bounds"][0][0], partition["bounds"][0][1]
        min_z, max_z = partition["bounds"][1][0], partition["bounds"][1][1]

        # Get a unique color for each partition
        rect = plt.Rectangle(
            (min_x, min_z), max_x - min_x, max_z - min_z, linewidth=1, edgecolor="black", facecolor="none"
        )
        ax.add_patch(rect)

        fig.savefig(os.path.join(save_path, f"{partition_id}.png"), dpi=200)
        plt.close(fig)

def read_camera_parameters(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    camera_extrinsics_list = []
    camera_intrinsics_list = []
    frames = data["frames"]
    for frame in frames:
        camera_intrinsics = np.array([
            [frame["fl_x"], 0, frame["cx"]],
            [0, frame["fl_y"], frame["cy"]],
            [0, 0, 1]
        ])
        camera_intrinsics_list.append(camera_intrinsics)
        
        # camera_extrinsics = np.linalg.inv(np.array(frame["transform_matrix"]))
        camera_extrinsics = np.array(frame["transform_matrix"])
        camera_extrinsics_list.append(camera_extrinsics)
    
    return camera_intrinsics_list, camera_extrinsics_list, frames

def save_json(path, cluster_frames):
    cluster_data = {
            "frames": cluster_frames
        }
    with open(path, 'w') as f:
        json.dump(cluster_data, f, indent=2)