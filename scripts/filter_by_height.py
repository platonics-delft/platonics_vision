import open3d as o3d
import numpy as np
import sys

def filter_by_height(pcd, z_low, z_high, x_low, x_high, y_low, y_high):
    # Filter the point cloud based on the z value
    pc = np.array(pcd.points)
    pcd_filtered = pcd.select_by_index(np.where(np.logical_and(pc[:,2] < z_high, pc[:,2] > z_low))[0])
    #pcd_filtered = pcd.select_by_index(np.where(np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.array(pcd.points)[:,2] > z_low, np.array(pcd.points)[:,2] < z_high), np.array(pcd.points)[:,0] > x_low), np.array(pcd.points)[:,0] < x_high), np.array(pcd.points)[:,1] > y_low), np.array(pcd.points)[:,1] < y_high)[0])
    return pcd_filtered

if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud(sys.argv[1])

    pcd_filtered = filter_by_height(pcd, z_low=0.08, z_high=0.2, x_low=-0.1, x_high=0.1, y_low=-0.1, y_high=0.1)
    print(f"Filtered point cloud has {len(pcd_filtered.points)} points")
    o3d.io.write_point_cloud(sys.argv[2], pcd_filtered)

