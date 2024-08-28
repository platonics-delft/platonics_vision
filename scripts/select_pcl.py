import sys
import open3d as o3d
import numpy as np
from shapely.geometry import Polygon, Point


def select_points_and_filter(pcd):
    # Visualizer for selecting points interactively
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # User picks points
    vis.destroy_window()

    # Get the indices of the selected points
    selected_indices = vis.get_picked_points()

    # If no points were selected, return an empty point cloud
    if len(selected_indices) == 0:
        print("No points were selected.")
        return o3d.geometry.PointCloud()

    # Extract the selected points
    selected_points = np.asarray(pcd.points)[selected_indices]

    # Create a polygon in the XY plane
    polygon = Polygon(selected_points[:, :2])

    # Filter the original points based on whether they are inside the polygon
    points = np.asarray(pcd.points)
    mask = np.array([polygon.contains(Point(x, y)) for x, y in points[:, :2]])


    # Filter the point cloud with the mask
    filtered_points = points[mask]

    # Create a new point cloud with the filtered points
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])

    return filtered_pcd

def main():
    # Load or create your point cloud
    pcd = o3d.io.read_point_cloud(sys.argv[1])

    # Select points and filter the point cloud
    filtered_pcd = select_points_and_filter(pcd)

    # Visualize the filtered point cloud
    if len(filtered_pcd.points) > 0:
        o3d.visualization.draw_geometries([filtered_pcd])
    else:
        print("No points within the selected region.")

    # save result
    o3d.io.write_point_cloud(sys.argv[2], filtered_pcd)

if __name__ == "__main__":
    main()

