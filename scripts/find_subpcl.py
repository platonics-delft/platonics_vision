import numpy as np
import open3d as o3d
import sys

def find_matching_subcloud_and_transformation(source_pcd, target_pcd, voxel_size=0.005, max_correspondence_distance=0.05):
    # Downsample the point clouds
    source_down = source_pcd.voxel_down_sample(voxel_size)
    target_down = target_pcd.voxel_down_sample(voxel_size)

    # Estimate normals for both point clouds
    source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2 * voxel_size, max_nn=30))
    target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2 * voxel_size, max_nn=30))

    # Compute FPFH features for source and target point clouds
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=5 * voxel_size, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=5 * voxel_size, max_nn=100))

    # Set up the RANSAC registration pipeline
    distance_threshold = max_correspondence_distance * 1.5
    ransac_n = 4
    checkers = [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
    ]
    criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)

    # Perform RANSAC registration
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=ransac_n,
        checkers=checkers,
        criteria=criteria
    )

    print("RANSAC Transformation:")
    print(result_ransac.transformation)

    # Refine the registration using ICP
    result_icp = o3d.pipelines.registration.registration_colored_icp(
        source_pcd, target_pcd, max_correspondence_distance=max_correspondence_distance,
        init=np.identity(4),
        #result_ransac.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())

    print("ICP Refinement Transformation:")
    print(result_icp.transformation)

    return result_ransac.transformation, target_pcd

# Example usage with synthetic data
if __name__ == "__main__":

    source_pcd = o3d.io.read_point_cloud(sys.argv[1])

    target_pcd = o3d.io.read_point_cloud(sys.argv[2])



    # Find the matching sub-cloud and the transformation
    transformation, matched_subcloud = find_matching_subcloud_and_transformation(source_pcd, target_pcd)


    # Print the result
    print("Estimated Transformation Matrix:")
    print(transformation)

    # draw the result
    source_pcd.transform(transformation)
    o3d.visualization.draw_geometries([source_pcd, target_pcd])


