import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation


def compute_bone_lengths(kpts, joints_hierarchy):
    """
    Iterate over joints with a parent and calculate length of each bone.
    This will help with visualization later.
    """
    bone_lengths = {}

    for joint, parents in joints_hierarchy.items():
        if not parents:
            continue  # skip root
        parent = parents[0]

        # Use only left-side or midline joints
        if "left" not in joint and joint not in ("spine",):
            continue

        dist = np.linalg.norm(kpts[joint] - kpts[parent], axis=1)
        bone_lengths[joint] = dist.mean()

    #add the right side joints as well
    opposite_lengths = {}
    for joint, b_length in bone_lengths.items():
        left_index = joint.find('left')
        if left_index != -1:
            #swap left -> right
            opposite_lengths['right' + joint[left_index + 4:]] = b_length
    bone_lengths.update(opposite_lengths)

    return bone_lengths


def animate_skeleton(kpts_root, joints_hierarchy, accumulated_rotations= None, interval=50):
    """
    Animates the skeleton over all frames using matplotlib.

    Args:
        kpts_root: dict[str, np.ndarray] - [num_frames, 3]
        joints_hierarchy: dict[str, list[str]] - ancestry, first element = direct parent
        bone_lengths: dict[str, float] (optional)
        interval: int - milliseconds between frames
    """
    num_frames = next(iter(kpts_root.values())).shape[0]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Prepare lines for each bone
    lines = {}
    for joint, parents in joints_hierarchy.items():
        if not parents:
            continue
        parent = parents[0]
        line, = ax.plot([], [], [], 'o-', lw=2, markersize=4, color='blue')
        lines[joint] = (line, parent)

    # Optional: add joint labels (fixed at origin, updated each frame)
    joint_texts = {}
    for joint in kpts_root.keys():
        txt = ax.text(0, 0, 0, joint, fontsize=8)
        joint_texts[joint] = txt

    # Set axis labels
    ax.set_xlabel('X (forward)')
    ax.set_ylabel('Y (left)')
    ax.set_zlabel('Z (up)')
    ax.set_box_aspect([1,1,1])
    ax.view_init(elev=20, azim=120)

    # Determine limits
    all_points = np.concatenate(list(kpts_root.values()), axis=0)
    ax.set_xlim(np.min(all_points[:,0]), np.max(all_points[:,0]))
    ax.set_ylim(np.min(all_points[:,1]), np.max(all_points[:,1]))
    ax.set_zlim(np.min(all_points[:,2]), np.max(all_points[:,2]))

    axis_lines = {}
    for joint in kpts_root.keys():
        line_x, = ax.plot([], [], [], '-', color='red', lw=2.5)
        line_y, = ax.plot([], [], [], '-', color='green', lw=2.5)
        line_z, = ax.plot([], [], [], '-', color='blue', lw=2.5)
        axis_lines[joint] = (line_x, line_y, line_z)

    # Base axis vectors (in the joint's local frame)
    axis_len = 1.0 # Example length
    local_x = np.array([axis_len, 0, 0])
    local_y = np.array([0, axis_len, 0])
    local_z = np.array([0, 0, axis_len])
    local_axes = np.stack([local_x, local_y, local_z], axis=0) # [3, 3]

    def update(frame):
        # update bone positions
        for joint, (line, parent) in lines.items():
            child_pos = kpts_root[joint][frame]
            parent_pos = kpts_root[parent][frame]
            line.set_data([child_pos[0], parent_pos[0]],
                          [child_pos[1], parent_pos[1]])
            line.set_3d_properties([child_pos[2], parent_pos[2]])

        # update joint labels
        for joint, txt in joint_texts.items():
            pos = kpts_root[joint][frame]
            txt.set_position((pos[0], pos[1]))
            txt.set_3d_properties(pos[2])

        if accumulated_rotations:
            # Update joint axes visualization
            for joint, (line_x, line_y, line_z) in axis_lines.items():

                if not joint in accumulated_rotations.keys(): continue

                pos = kpts_root[joint][frame]
                
                # 1. Get the rotation for the current joint and frame
                R_quat = accumulated_rotations[joint][frame]
                R_obj = Rotation.from_quat(R_quat)
                
                # 2. Transform the local axis vectors to the Root Frame
                # R_obj.apply takes vectors from the frame R_obj defines (the local frame) 
                # and outputs them in the base frame (the Root Frame)
                rotated_axes = R_obj.apply(local_axes) # [3, 3] matrix of vectors

                # X-axis (Red)
                x_end = pos + rotated_axes[0]
                line_x.set_data([pos[0], x_end[0]], [pos[1], x_end[1]])
                line_x.set_3d_properties([pos[2], x_end[2]])

                # Y-axis (Green)
                y_end = pos + rotated_axes[1]
                line_y.set_data([pos[0], y_end[0]], [pos[1], y_end[1]])
                line_y.set_3d_properties([pos[2], y_end[2]])

                # Z-axis (Blue)
                z_end = pos + rotated_axes[2]
                line_z.set_data([pos[0], z_end[0]], [pos[1], z_end[1]])
                line_z.set_3d_properties([pos[2], z_end[2]])

        all_lines = [l[0] for l in lines.values()] + [item for sublist in axis_lines.values() for item in sublist]
        return all_lines + list(joint_texts.values())

    ani = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=False)
    plt.show()

def smooth_keypoints(kpts_dict, kernel_size=3):
    """
    Apply a temporal median filter to keypoints to reduce noise.

    Args:
        kpts_dict: dict[str, np.ndarray] - [num_frames, 3]
        kernel_size: int - size of median filter window (should be odd)

    Returns:
        dict[str, np.ndarray] - smoothed keypoints
    """
    smoothed = {}
    for joint, pos in kpts_dict.items():
        # Apply median filter along the time axis (axis=0) for each coordinate
        smoothed[joint] = median_filter(pos, size=(kernel_size, 1), mode='nearest')
    return smoothed

def get_children(joints_hierarchy):
    """
    Returns a dictionary mapping each joint to its direct children.
    """
    children = {joint: [] for joint in joints_hierarchy}

    for joint, parents in joints_hierarchy.items():
        if parents:  # has a parent
            parent = parents[0]  # direct parent
            children[parent].append(joint)

    return children


def smooth_quaternion_rotations_rotvec(local_joint_rots, window_size=5):
    """
    Smoothes an array of quaternions by converting them to rotation vectors,
    applying a moving average filter, and converting them back.

    Parameters:
    - local_joint_rots (np.ndarray): Array of quaternions, shape (N, 4).
    - window_size (int): Size of the moving average window (must be odd).

    Returns:
    - np.ndarray: Array of smoothed quaternions, shape (N, 4).
    """
    
    # 1. Convert Quaternions to Rotation Objects
    # Ensure input is an array of (x, y, z, w) quaternions
    rotation_objects = Rotation.from_quat(local_joint_rots)
    
    # 2. Convert Rotation Objects to Rotation Vectors (rot_vec)
    # The rot_vec is a 3D vector: magnitude is the angle, direction is the axis (theta*u)
    rot_vecs = rotation_objects.as_rotvec()  # Shape (N, 3)
    
    # Check for valid window size
    if window_size % 2 == 0:
        window_size += 1  # Ensure window is odd for centered smoothing
        print(f"Warning: Window size adjusted to {window_size} for centering.")

    # Calculate padding size for the moving average
    pad_size = window_size // 2
    
    # 3. Apply Moving Average Filter to Rotation Vectors
    # This is a linear filter that works well on the continuous rot_vec space
    
    N = len(rot_vecs)
    smoothed_rot_vecs = np.zeros_like(rot_vecs, dtype=np.float32)
    
    # Pad the rotation vectors for boundary handling
    # We use 'edge' mode to repeat the start/end values
    padded_rot_vecs = np.pad(rot_vecs, ((pad_size, pad_size), (0, 0)), mode='edge')
    
    for i in range(N):
        # The window starts 'pad_size' frames before 'i' in the padded array
        # and ends 'pad_size' frames after 'i'
        start_idx = i
        end_idx = i + window_size
        
        # Calculate the mean (moving average) over the window
        window = padded_rot_vecs[start_idx:end_idx]
        smoothed_rot_vecs[i] = np.mean(window, axis=0)
        
    # 4. Convert Smoothed Rotation Vectors back to Quaternions
    smoothed_rotation_objects = Rotation.from_rotvec(smoothed_rot_vecs)
    
    # Return the final array of smoothed quaternions (x, y, z, w)
    return smoothed_rotation_objects.as_quat()

def fix_sign_flipping(joint_rots):

    q_fixed = np.copy(joint_rots)

    for i in range(1, len(q_fixed)):
        # Calculate the dot product between current and previous frame
        dot_product = np.dot(q_fixed[i], q_fixed[i-1])
        
        # If dot product is negative, flip the sign of the current quaternion
        if dot_product < 0:
            q_fixed[i] *= -1

    return q_fixed

#animate the joint rotations in root frame
def animate_joint_rotations(joint_rots, joints_heirarchy, joints_offsets, bone_lengths):

    num_frames = len(joint_rots[list(joint_rots.keys())[0]])
    
    #hold the accumulated rotations for each joint
    accumulated_rotations = {'hip': np.tile([0,0,0,1], (num_frames, 1))}

    #get the children of each joint
    children = get_children(joints_heirarchy)

    #keypoints in root frame
    keypoints = {'hip': np.tile([0,0,0], (num_frames, 1))}
    max_depth = np.max([len(joints) for joints in joints_heirarchy.values()])

    for depth in range(max_depth):

        if depth == 0:
            #for the hip joint, the children can be directly written
            for child in children['hip']:
                locs = np.tile(joints_offsets[child], (num_frames, 1)) * np.array(bone_lengths[child])
                keypoints[child] = locs
            continue

        #calculate only at current depth
        for joint, parents in joints_heirarchy.items():
            if len(parents) != depth: continue
            #skip endpoints
            if len(children[joint]) == 0: continue

            #parent's rotation
            parent = joints_heirarchy[joint][0] #direct parent
            R_parent = accumulated_rotations[parent]

            R_joint = Rotation.from_quat(R_parent) * Rotation.from_quat(joint_rots[joint])
            accumulated_rotations[joint] = R_joint.as_quat()

            parent_pos = keypoints[joint]

            for child in children[joint]:
                #rotate the bone
                locs = R_joint.apply(np.tile(joints_offsets[child], (num_frames, 1)) * np.array(bone_lengths[child]))
                #find the position of the keypoint
                keypoints[child] = parent_pos + locs

    animate_skeleton(keypoints, joints_heirarchy)