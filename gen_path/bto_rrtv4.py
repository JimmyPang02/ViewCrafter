"""
v3: 实现生成相机位姿序列的功能（切线or始终看中间）
"""
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def generateRandomPoint(ptCloud):
    # scale factor
    scale_factor = 0.5

    # Get point cloud boundaries
    xyz = np.asarray(ptCloud.points)
    x_min, y_min, z_min = xyz.min(axis=0)
    x_max, y_max, z_max = xyz.max(axis=0)

    # Calculate scaled boundary ranges
    x_range = (x_max - x_min) * scale_factor
    y_range = (y_max - y_min) * scale_factor
    z_range = (z_max - z_min) * scale_factor

    # Calculate scaled boundary centers
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2

    # Generate random point
    x = x_center + (x_range / 2) * (2 * np.random.rand() - 1)
    y = y_center + (y_range / 2) * (2 * np.random.rand() - 1)
    z = z_center + (z_range / 2) * (2 * np.random.rand() - 1)

    point = np.array([x, y, z])
    return point



def checkpoint(point, ptCloud, params, ptCloud_tree):
    K = 10
    node_check = False  # Safe by default
    
    # 确保 point 是正确的格式
    point = np.asarray(point, dtype=np.float64)  # 转换为 float64
    if point.size != 3:
        return True  # 如果点的维度不正确，认为是不安全的
    
    try:
        # 确保点是一个 3D 向量
        search_point = point.reshape(3,)
        [k, idx, dists] = ptCloud_tree.search_knn_vector_3d(search_point, K)
        
        # 检查返回值
        if k <= 0:
            return True
            
        dists = np.sqrt(dists)
        for dist in dists:
            if dist <= params['safe_dist']:
                node_check = True  # Unsafe
                break
                
    except Exception as e:
        print(f"KNN search error: {e}")
        return True  # 如果搜索失败，认为是不安全的
        
    return node_check


def Collisiondetect(node1, node2, ptCloud, sizemax, params, ptCloud_tree):
    collision_flag = False
    x, y, z = node1[0], node1[1], node1[2]
    if (x > sizemax['x']) or (x < 0) or (y > sizemax['y']) or (y < 0):
        collision_flag = True
    else:
        for sigma in np.arange(0, 1.1, 0.1):
            p = sigma * node1[0:3] + (1 - sigma) * node2[0:3]
            if checkpoint(p, ptCloud, params, ptCloud_tree):
                collision_flag = True
                break
    return collision_flag


def extendTree(RRTnode, goal, stepsize, ptCloud, sizemax, params, ptCloud_tree):
    flag1 = False
    qet = True
    max_attempts = 100  # Maximum number of attempts to find a valid extension
    attempts = 0

    while not flag1 and attempts < max_attempts:
        attempts += 1
        if not qet:
            # q_rand = generateRandomPoint(ptCloud)
            q_rand = np.array([np.random.rand() * sizemax['x'],
                               np.random.rand() * sizemax['y'],
                               np.random.rand() * sizemax['h']])
        else:
            q_rand = goal[0:3]

        # Find the node in RRTnode that is closest to q_rand
        diffs = RRTnode[:, 0:3] - q_rand
        dists = np.linalg.norm(diffs, axis=1)
        idx = np.argmin(dists)
        nearest_node = RRTnode[idx]

        # Create a new node in the direction of q_rand
        direction = q_rand - nearest_node[0:3]
        if np.linalg.norm(direction) == 0:
            direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        q_new = nearest_node[0:3] + direction * stepsize

        parent_cost = nearest_node[4]  # cost is at index 4
        cost_new = parent_cost + np.linalg.norm(q_new - nearest_node[0:3])
        new_node = np.array([q_new[0], q_new[1], q_new[2], 0, cost_new, idx])

        if not Collisiondetect(new_node, nearest_node, ptCloud, sizemax, params, ptCloud_tree):
            # Add new node to the tree
            RRTnode = np.vstack((RRTnode, new_node))
            flag1 = True
        else:
            qet = False
            flag1 = False

    if not flag1:
        # Could not find a valid extension after max_attempts
        return RRTnode, False

    # Check if new node connects directly to goal
    if (np.linalg.norm(new_node[0:3] - goal[0:3]) < stepsize) and \
       (not Collisiondetect(new_node, goal, ptCloud, sizemax, params, ptCloud_tree)):
        flag = True
        RRTnode[-1, 3] = 1  # Mark node as connecting to end
    else:
        flag = False

    return RRTnode, flag


def findMinimumPath(tree):
    # tree is an array with columns: [x, y, z, flag, cost, parent_index]
    # Find nodes where flag == 1
    connectingNodes = tree[tree[:, 3] == 1]
    if connectingNodes.size == 0:
        return None  # No path found
    # Find the node with minimum cost
    idx = np.argmin(connectingNodes[:, 4])
    min_node = connectingNodes[idx]
    path = [min_node]
    parent_idx = int(min_node[5])
    while parent_idx != -1:
        parent_node = tree[parent_idx]
        path.insert(0, parent_node)
        parent_idx = int(parent_node[5])
    # Convert list of nodes to 2D NumPy array
    path = np.vstack(path)
    return path


def downsample(path, ptCloud, sizemax, params, ptCloud_tree):
    path_l = len(path)
    dis_opt = np.zeros(path_l)
    path_start = path[0, 0:3]
    indx = [0]
    for i in range(1, path_l):
        dis_opt[i] = np.linalg.norm(path_start - path[i, 0:3])
        if (dis_opt[i] > dis_opt[i-1]) or (dis_opt[i] > 0):
            if Collisiondetect(path_start, path[i, 0:3], ptCloud, sizemax, params, ptCloud_tree):
                path_start = path[i-1, 0:3]
                indx.append(i-1)
    indx.append(path_l - 1)
    path_opt = [path[0, 0:3]]
    for i in range(1, len(indx)):
        path_opt.append(path[indx[i], 0:3])
    path_opt = np.array(path_opt)
    return path_opt


def upsample(path, ptCloud, sizemax, params, ptCloud_tree):
    P = path.T
    m = P.shape[1]
    l = np.zeros(m)
    for k in range(1, m):
        l[k] = np.linalg.norm(P[:, k] - P[:, k - 1]) + l[k - 1]
    iter = 1
    while iter <= 1000:
        s1 = np.random.rand() * l[-1]
        s2 = np.random.rand() * l[-1]
        if s2 < s1:
            s1, s2 = s2, s1
        for k in range(1, m):
            if s1 < l[k]:
                i = k - 1
                break
        for k in range(i + 1, m):
            if s2 < l[k]:
                j = k - 1
                break
        if j <= i:
            iter += 1
            continue
        t1 = (s1 - l[i]) / (l[i + 1] - l[i])
        gamma1 = (1 - t1) * P[:, i] + t1 * P[:, i + 1]
        t2 = (s2 - l[j]) / (l[j + 1] - l[j])
        gamma2 = (1 - t2) * P[:, j] + t2 * P[:, j + 1]
        col1 = Collisiondetect(P[:, i], gamma1, ptCloud, sizemax, params, ptCloud_tree)
        if col1:
            iter += 1
            continue
        col2 = Collisiondetect(gamma1, gamma2, ptCloud, sizemax, params, ptCloud_tree)
        if col2:
            iter += 1
            continue
        col3 = Collisiondetect(P[:, j], gamma2, ptCloud, sizemax, params, ptCloud_tree)
        if col3:
            iter += 1
            continue
        col4 = Collisiondetect(P[:, j], P[:, i], ptCloud, sizemax, params, ptCloud_tree)
        if col4:
            iter += 1
            continue
        P = np.hstack((P[:, :i + 1], gamma1.reshape(-1, 1), gamma2.reshape(-1, 1), P[:, j + 1:]))
        m = P.shape[1]
        l = np.zeros(m)
        for k in range(1, m):
            l[k] = np.linalg.norm(P[:, k] - P[:, k - 1]) + l[k - 1]
        iter += 1
    return P


def visualize_initial_scene(ptCloud, q_init, q_goal, boundaries, cloud_scale_max):
    """
    可视化初始场景，包括点云、边界框、起点和终点
    
    Args:
        ptCloud: 输入的点云数据
        q_init: 起始点位置
        q_goal: 目标点位置
        boundaries: 包含边界信息的字典 {'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'}
        cloud_scale_max: 点云的最大尺度
    """
    print("\n正在可视化初始点云和起始点位置...")
    initial_geometries = []
    
    # 添加点云
    initial_geometries.append(ptCloud)
    
    # 计算并添加XY平面中心点（黄色）
    xy_center = np.array([
        (boundaries['x_min'] + boundaries['x_max']) / 2,
        (boundaries['y_min'] + boundaries['y_max']) / 2,
        boundaries['z_max']  # 在最高z高度的XY平面上
    ])
    center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=cloud_scale_max * 0.02)
    center_sphere.translate(xy_center)
    center_sphere.paint_uniform_color([1, 1, 0])  # 黄色
    initial_geometries.append(center_sphere)
    
    # 打印中心点坐标
    print(f"\nXY平面中心点坐标（最高处）: ({xy_center[0]:.2f}, {xy_center[1]:.2f}, {xy_center[2]:.2f})")
    
    # 添加边界点（8个顶点）
    boundary_points = [
        [boundaries['x_min'], boundaries['y_min'], boundaries['z_min']],  # 前下左
        [boundaries['x_max'], boundaries['y_min'], boundaries['z_min']],  # 前下右
        [boundaries['x_min'], boundaries['y_max'], boundaries['z_min']],  # 后下左
        [boundaries['x_max'], boundaries['y_max'], boundaries['z_min']],  # 后下右
        [boundaries['x_min'], boundaries['y_min'], boundaries['z_max']],  # 前上左
        [boundaries['x_max'], boundaries['y_min'], boundaries['z_max']],  # 前上右
        [boundaries['x_min'], boundaries['y_max'], boundaries['z_max']],  # 后上左
        [boundaries['x_max'], boundaries['y_max'], boundaries['z_max']]   # 后上右
    ]
    
    # 创建边界点的可视化（绿色球体）
    for point in boundary_points:
        boundary_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=cloud_scale_max * 0.015)
        boundary_sphere.translate(point)
        boundary_sphere.paint_uniform_color([0, 1, 0])  # 绿色
        initial_geometries.append(boundary_sphere)
        
    # 创建边界框线条
    lines = [
        [0, 1], [1, 3], [3, 2], [2, 0],  # 底面
        [4, 5], [5, 7], [7, 6], [6, 4],  # 顶面
        [0, 4], [1, 5], [2, 6], [3, 7]   # 竖直边
    ]
    colors = [[0, 1, 0] for _ in range(len(lines))]  # 绿色线条
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(boundary_points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    initial_geometries.append(line_set)
    
    # 添加起点（蓝色球体）
    start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=cloud_scale_max * 0.01)
    start_sphere.translate(q_init[0:3])
    start_sphere.paint_uniform_color([0, 0, 1])  # 蓝色
    initial_geometries.append(start_sphere)
    
    # 添加终点（红色球体）
    goal_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=cloud_scale_max * 0.01)
    goal_sphere.translate(q_goal[0:3])
    goal_sphere.paint_uniform_color([1, 0, 0])  # 红色
    initial_geometries.append(goal_sphere)
    
    # 可视化初始场景
    o3d.visualization.draw_geometries(initial_geometries)


def main(path=None):
    try:
        # Load point cloud
        print("开始加载点云...")
        # Load point cloud
        if path is None:
            ptCloud = o3d.io.read_point_cloud('scene.ply')
        else:
            ptCloud = o3d.io.read_point_cloud(path)

        if not ptCloud.has_points():
            raise ValueError("Point cloud is empty or file could not be opened.")

        # Create KDTree for nearest neighbor search
        ptCloud_tree = o3d.geometry.KDTreeFlann(ptCloud)

        # Get point cloud boundaries
        xyz = np.asarray(ptCloud.points)
        x_min, y_min, z_min = xyz.min(axis=0)
        x_max, y_max, z_max = xyz.max(axis=0)
        sizemax = {'x': x_max, 'y': y_max, 'h': z_max}
        
        # 存储边界信息
        boundaries = {
            'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max,
            'z_min': z_min, 'z_max': z_max
        }

        # Compute cloud scale
        cloud_scale_x = x_max - x_min
        cloud_scale_y = y_max - y_min
        cloud_scale_z = z_max - z_min
        cloud_scale_max = max(cloud_scale_x, cloud_scale_y, cloud_scale_z)
        cloud_scale_min = min(cloud_scale_x, cloud_scale_y, cloud_scale_z)

        # Set parameters based on cloud scale
        params = {}
        params['safe_dist'] = cloud_scale_max * 0.05 # 0.01
        params['stepsize'] = cloud_scale_max * 0.01
        params['threshold'] = cloud_scale_max * 0.05
        params['sample_density'] = 0.05

        # Initialize start and goal positions
        random_goal = generateRandomPoint(ptCloud)
        q_init = np.array([
            (x_max + x_min) / 2,  # x中心
            (y_max + y_min) / 2,  # y中心
            z_max,                # 最高处z
            0, 0, -1
        ])
        q_goal = np.array([random_goal[0], random_goal[1], random_goal[2], 1, 0, -1])

        # Ensure initial and goal positions are within valid range
        q_init[0:3] = np.clip(q_init[0:3], 0, [sizemax['x'], sizemax['y'], sizemax['h']])
        q_goal[0:3] = np.clip(q_goal[0:3], 0, [sizemax['x'], sizemax['y'], sizemax['h']])

        # Ensure initial and goal positions are not inside obstacles
        start_safe = not checkpoint(q_init[0:3], ptCloud, params, ptCloud_tree)
        goal_safe = not checkpoint(q_goal[0:3], ptCloud, params, ptCloud_tree)

        print("\n安全性检查:")
        print(f"起点是否安全: {start_safe}")
        print(f"终点是否安全: {goal_safe}")
        print(f"安全距离参数: {params['safe_dist']}")

        # 调用可视化函数
        visualize_initial_scene(ptCloud, q_init, q_goal, boundaries, cloud_scale_max)

        while checkpoint(q_init[0:3], ptCloud, params, ptCloud_tree) or checkpoint(q_goal[0:3], ptCloud, params, ptCloud_tree):
            if checkpoint(q_goal[0:3], ptCloud, params, ptCloud_tree):
                q_goal[0:3] = generateRandomPoint(ptCloud)

        stepsize = params['stepsize']
        RRTnode = q_init.reshape(1, -1)
        RRTnode1 = q_goal.reshape(1, -1)

        # Main loop to extend trees and find path
        path_found = 0
        max_iterations = 1000  # Maximum number of iterations for the main loop
        iteration = 0

        while path_found < 1 and iteration < max_iterations:
            iteration += 1
            RRTnode, flag = extendTree(RRTnode, q_goal, stepsize, ptCloud, sizemax, params, ptCloud_tree)
            RRTnode1, flag1 = extendTree(RRTnode1, RRTnode[-1], stepsize, ptCloud, sizemax, params, ptCloud_tree)
            path_found += flag1

        if path_found < 1:
            print("No path found within the maximum number of iterations.")
            return

        # Manually set flags as in the original MATLAB code
        RRTnode[-1, 3] = 1  # Mark last node in RRTnode as connecting to goal
        RRTnode1[0, 3] = 0  # Ensure first node in RRTnode1 is not marked

        # Find and combine paths from both trees
        path1 = findMinimumPath(RRTnode) # 可能是这
        path2 = findMinimumPath(RRTnode1)

        # Ensure paths are not None
        if path1 is None or path2 is None:
            print("No path found")
            return

        # Print shapes for debugging
        print("path1 shape:", path1.shape)
        print("path2 shape:", path2.shape)

        # Reshape if paths are 1D arrays
        if path1.ndim == 1:
            path1 = path1.reshape(1, -1)
        if path2.ndim == 1:
            path2 = path2.reshape(1, -1)

        # Reverse path2 and stack with path1
        path2_reversed = np.flip(path2, axis=0)
        path = np.vstack((path1, path2_reversed))

        # Downsample and upsample paths
        path_opt = downsample(path, ptCloud, sizemax, params, ptCloud_tree)
        if path_opt is not None:
            P = upsample(path_opt, ptCloud, sizemax, params, ptCloud_tree)

        # Create geometries for visualization
        geometries = []

        # Add point cloud
        geometries.append(ptCloud)

        # Add original path (red)
        if path is not None and path.size > 0:
            lines = [[i, i + 1] for i in range(len(path) - 1)]
            colors = [[1, 0, 0] for _ in lines]  # Red color
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(path[:, 0:3]),
                lines=o3d.utility.Vector2iVector(lines),
            )
            line_set.colors = o3d.utility.Vector3dVector(colors)
            geometries.append(line_set)

        # Add optimized path (yellow)
        if path_opt is not None and path_opt.size > 0:
            lines = [[i, i + 1] for i in range(len(path_opt) - 1)]
            colors = [[1, 1, 0] for _ in lines]  # Yellow color
            line_set_opt = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(path_opt[:, 0:3]),
                lines=o3d.utility.Vector2iVector(lines),
            )
            line_set_opt.colors = o3d.utility.Vector3dVector(colors)
            geometries.append(line_set_opt)

        # Add upsampled path (green)
        if 'P' in locals() and P.size > 0:
            P_T = P.T
            lines = [[i, i + 1] for i in range(P_T.shape[0] - 1)]
            colors = [[0, 1, 0] for _ in lines]  # Green color
            line_set_P = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(P_T[:, 0:3]),
                lines=o3d.utility.Vector2iVector(lines),
            )
            line_set_P.colors = o3d.utility.Vector3dVector(colors)
            geometries.append(line_set_P)

        # Add smoothed path (blue)
        # For simplicity, we can use the upsampled path as the smoothed path
        # Alternatively, you can perform spline fitting using scipy and sample points
        if 'P' in locals() and P.size > 0:
            path_up = P.T
            keypoints = downsample(path_up, ptCloud, sizemax, params, ptCloud_tree)
            if keypoints is not None and keypoints.shape[0] > 1:
                from scipy.interpolate import splprep, splev
                m = keypoints.shape[0]
                k = min(3, m - 1)
                if k >= 1:
                    print(f"Number of keypoints: {m}, spline degree: {k}")
                    tck, u = splprep([keypoints[:, 0], keypoints[:, 1], keypoints[:, 2]], s=0, k=k)
                    u_fine = np.linspace(0, 1, num=1000)
                    x_spline, y_spline, z_spline = splev(u_fine, tck)
                    spline_points = np.vstack((x_spline, y_spline, z_spline)).T
                    lines = [[i, i + 1] for i in range(len(spline_points) - 1)]
                    colors = [[0, 0, 1] for _ in lines]  # Blue color
                    line_set_spline = o3d.geometry.LineSet(
                        points=o3d.utility.Vector3dVector(spline_points),
                        lines=o3d.utility.Vector2iVector(lines),
                    )
                    line_set_spline.colors = o3d.utility.Vector3dVector(colors)
                    geometries.append(line_set_spline)
                else:
                    print("Not enough keypoints to generate a spline curve.")
            else:
                print("Keypoints are not sufficient for spline fitting.")


        # **Compute camera poses along the spline path**
        camera_poses = []
        if 'spline_points' in locals() and spline_points.shape[0] > 1:
            # Compute derivatives along the spline
            x_der, y_der, z_der = splev(u_fine, tck, der=1)
            tangents = np.vstack((x_der, y_der, z_der)).T
            # Normalize tangents
            tangents_norm = np.linalg.norm(tangents, axis=1, keepdims=True)
            tangents_normalized = tangents / tangents_norm

            # For visualization, create coordinate frames at intervals
            num_frames = 50  # Adjust the number of frames as needed
            indices = np.linspace(0, len(spline_points) - 1, num_frames).astype(int)

            for idx in indices:
                position = spline_points[idx]
                tangent = tangents_normalized[idx]

                # Create rotation matrix aligning z-axis with tangent vector
                # Use arbitrary up vector, e.g., [0, 0, 1], and compute right vector
                up = np.array([0, 0, 1])
                if np.allclose(tangent, up) or np.allclose(tangent, -up):
                    up = np.array([1, 0, 0])  # Choose a different up vector if tangent is close to [0,0,1]

                right = np.cross(tangent, up)
                right /= np.linalg.norm(right)
                up_corrected = np.cross(right, tangent)

                rotation_matrix = np.column_stack((right, up_corrected, tangent))

                # Create a coordinate frame at this position
                mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=cloud_scale_max * 0.05)
                mesh_frame.translate(position)
                mesh_frame.rotate(rotation_matrix, center=position)
                geometries.append(mesh_frame)

                # Store camera pose (position and orientation)
                camera_pose = {
                    'position': position,
                    'orientation': rotation_matrix
                }
                camera_poses.append(camera_pose)

        # **Optionally, save camera poses to a file or use them as needed**
        # For example, save to a JSON file
        import json
        camera_poses_serializable = []
        for pose in camera_poses:
            camera_poses_serializable.append({
                'position': pose['position'].tolist(),
                'orientation': pose['orientation'].tolist()
            })
        with open('camera_poses.json', 'w') as f:
            json.dump(camera_poses_serializable, f, indent=4)

        # vis
        # Add start and goal points
        start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=cloud_scale_max * 0.01)
        start_sphere.translate(q_init[0:3])
        start_sphere.paint_uniform_color([0, 0, 1])  # Blue color
        geometries.append(start_sphere)

        goal_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=cloud_scale_max * 0.01)
        goal_sphere.translate(q_goal[0:3])
        goal_sphere.paint_uniform_color([1, 0, 0])  # Red color
        geometries.append(goal_sphere)

        # Visualize all geometries
        o3d.visualization.draw_geometries(geometries)


    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
