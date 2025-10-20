import sys
import numpy as np
from scipy.spatial.transform import Rotation
import utils
import matplotlib.pyplot as plt
import pickle

if len(sys.argv) != 2:
    print('Call the program with keypoints data.')
    quit()


#this dictionary defines how each joint relates to the root joint.
#in other words, the entries in the arrays are the parents of the joint.
#the hip joint is assigned to be the root joint
joints_heirarchy = {
    'hip': [],
    'left_waist': ['hip'],
    'left_knee': ['left_waist', 'hip'],
    'left_ankle': ['left_knee', 'left_waist', 'hip'],
    'right_waist': ['hip'],
    'right_knee': ['right_waist', 'hip'],
    'right_ankle': ['right_knee', 'right_waist', 'hip'],
    'spine': ['hip'],
    'left_shoulder': ['spine', 'hip'],
    'left_elbow': ['left_shoulder', 'spine', 'hip'],
    'left_wrist': ['left_elbow', 'left_shoulder', 'spine', 'hip'],
    'right_shoulder': ['spine', 'hip'],
    'right_elbow': ['right_shoulder', 'spine', 'hip'],
    'right_wrist': ['right_elbow', 'right_shoulder', 'spine', 'hip'],
}

#To calculate the joint angles, a T pose is defined.
#Each joint has x pointing forward and z pointing up. 
#The offsets are defined from the direct parent axes. 
joints_offsets = {
    'left_waist': [0,1,0],
    'left_knee': [0,0,-1],
    'left_ankle': [0,0,-1],
    'right_waist': [0,-1,0],
    'right_knee': [0,0,-1],
    'right_ankle': [0,0,-1],
    'spine': [0,0,1],
    'left_shoulder': [0,1,0],
    'left_elbow': [0,1,0],
    'left_wrist': [0,1,0],
    'right_shoulder': [0,-1,0],
    'right_elbow': [0,-1,0],
    'right_wrist': [0,-1,0]
}

# convert data to dictionary
def to_dictionary(kpts):

    #these are the indices of keypoints.
    keypoints_inds = {
        'left_shoulder': 0,
        'right_shoulder': 1,
        'left_elbow': 2,
        'right_elbow': 3,
        'left_wrist': 4,
        'right_wrist': 5,
        'left_waist': 6,
        'right_waist': 7,
        'left_knee': 8,
        'right_knee': 9,
        'left_ankle': 10,
        'right_ankle': 11
    }

    kpts = np.array(kpts).reshape(-1, len(keypoints_inds.keys()), 3)
    
    kpts_dict = {}
    for kpt, ind in keypoints_inds.items():
        kpts_dict[kpt] = kpts[:, ind, :]
    
    return kpts_dict


def add_hip_spine(kpts_dict):
    kpts_dict['hip'] = (kpts_dict['left_waist'] + kpts_dict['right_waist']) / 2
    kpts_dict['spine'] = (kpts_dict['left_shoulder'] + kpts_dict['right_shoulder']) / 2
    return kpts_dict


def normalize(v): 
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8, None)


def to_root_frame(kpts):
    """
    Convert mocap keypoints to the root (hip) frame.
    """

    root_pos = kpts["hip"]
    z_axis = kpts["spine"] - kpts["hip"]        #up
    y_axis = kpts["left_waist"] - kpts["hip"]   #left
    x_axis = np.cross(y_axis, z_axis)           #forward

    # Normalize and orthogonalize
    x_axis = normalize(x_axis)
    y_axis = normalize(np.cross(z_axis, x_axis))
    z_axis = normalize(np.cross(x_axis, y_axis))

    # Stack into rotation matrix
    R_root_mats = np.stack([x_axis, y_axis, z_axis], axis=-1)  # [num_frames, 3, 3]
    R_root = Rotation.from_matrix(R_root_mats)

    # Convert all keypoints into root frame
    kpts_root = {}
    for name, pos in kpts.items():
        rel = pos - root_pos
        kpts_root[name] = R_root.inv().apply(rel)

    return kpts_root, root_pos, R_root


def get_parent_rotation(target_joint, joints_heirarchy, joints_rotations):

    """
    Returns the rotation of the parent frame, which needs to be composed from the grandparents.
    Rotation of a joint depends on the configuration of all of its parent joints, which is applied recursively here.
    """
    parents = joints_heirarchy[target_joint]

    #compose the parent's rotation    
    rot = Rotation.identity()
    for p in parents[::-1]:
        rot = rot * Rotation.from_quat(joints_rotations[p])

    return rot


def construct_frames(keypoints, joint, joint_heirarchy, joint_rotations):
    """
    Unfortunately, we need to manually define stable secondary axis for each frame.
    This will allow us to calculate joint rotations without axis randomly flipping.
    """

    #get the parent's rotation
    R_parent = get_parent_rotation(joint, joints_heirarchy, joint_rotations)

    if joint == 'left_waist':
        primary_vec = keypoints['left_knee'] - keypoints['left_waist']
        constraint_vec = keypoints['hip'] - keypoints['left_waist']

        Z_expected = np.array([0, 0, -1])
        Y_expected = np.array([0, -1, 0])
        X_expected = np.cross(Y_expected, Z_expected)
    
    elif joint == 'right_waist':
        primary_vec = keypoints['right_knee'] - keypoints['right_waist']
        constraint_vec = keypoints['hip'] - keypoints['right_waist']

        Z_expected = np.array([0, 0, -1])
        Y_expected = np.array([0, 1, 0])
        X_expected = np.cross(Y_expected, Z_expected)

    elif joint == 'spine':
        primary_vec = keypoints['hip'] - keypoints['spine']
        constraint_vec = keypoints['right_shoulder'] - keypoints['left_shoulder']

        Z_expected = np.array([0, 0, -1])
        Y_expected = np.array([0, -1, 0])
        X_expected = np.cross(Y_expected, Z_expected)

    elif joint == 'left_shoulder':

        primary_vec = keypoints['left_elbow'] - keypoints['left_shoulder'] #spine to shoulder is rigid, so can just copy this
        constraint_vec = keypoints['spine'] - keypoints['hip']

        Z_expected = np.array([0, 1, 0])
        Y_expected = np.array([0, 0, 1])
        X_expected = np.cross(Y_expected, Z_expected)

    elif joint == 'right_shoulder':
        primary_vec = keypoints['right_elbow'] - keypoints['right_shoulder'] #spine to shoulder is rigid, so can just copy this
        constraint_vec = keypoints['spine'] - keypoints['hip']

        Z_expected = np.array([0, -1, 0])
        Y_expected = np.array([0, 0, 1])
        X_expected = np.cross(Y_expected, Z_expected)
    else:
        raise RuntimeError(f'Unkown joint name: {joint}')


    # Ortogonalize
    Z_current_raw = R_parent.inv().apply(normalize(primary_vec))
    C_raw = R_parent.inv().apply(normalize(constraint_vec))

    # primary axies
    Z_local = Z_current_raw

    # intermediate axis
    X_local_temp = np.cross(Z_local, C_raw)
    X_local_temp = normalize(X_local_temp)
    
    # final axis
    Y_local = np.cross(X_local_temp, Z_local)
    Y_local = normalize(Y_local)

    # recalculate to ensure orthogonality
    X_local = np.cross(Y_local, Z_local)

    # constuct rotation matrices
    R_current = np.stack([X_local, Y_local, Z_local], axis = -1)
    R_expected = np.stack([X_expected, Y_expected, Z_expected], axis = -1)
    
    return R_current, R_expected


def calculate_joint_angles(keypoints, joints_heirarchy, joints_offsets, children):
    
    """
    Calculate the joint angles frame by frame.    
    """

    #the longest joints chain in the data
    max_depth = np.max([len(joints) for joints in joints_heirarchy.values()])

    #calculate the number of frames
    num_frames = len(keypoints['hip'])

    #by default, the joint angles are calculated in root frame, which always has identity rotation
    joint_rotations = {'hip': np.tile(np.array([0,0,0,1], dtype=np.float32), (num_frames, 1))}

    #calculate the joint rotations from the smallest depth.
    for depth in range(1, max_depth): #skip root depth
        
        #calculate only at current depth
        for joint, parents in joints_heirarchy.items():
            if len(parents) != depth: continue
                   
            #skip endpoints
            if len(children[joint]) == 0: continue

            # some joints have a well behaved contraint axis.
            if joint in ['left_waist', 'right_waist', 'spine', 'left_shoulder', 'right_shoulder']:
                R_current, R_expected = construct_frames(keypoints, joint, joints_heirarchy, joint_rotations)

                # R_expected is the T-pose frame, R_current is the observed frame
                R_expected_inv = R_expected.transpose() # Inverse of an orthogonal matrix is its transpose
                R_local_joint_matrix = R_current @ R_expected_inv
                
                # Convert the final matrix to a quaternion
                R_quat = Rotation.from_matrix(R_local_joint_matrix).as_quat()
                local_joint_rots = np.array(R_quat)

            # these joints don't have a constraint axis. So just treat them as 1D rotation
            elif joint in ['left_knee', 'right_knee', 'left_elbow', 'right_elbow']:
                
                #get the parent's rotation. This is a rotation object with: [num_frames, rotations]
                R_parent = get_parent_rotation(joint, joints_heirarchy, joint_rotations)

                #calculate the joint angles
                child_joint = children[joint][0] #take the first child if there are multiple.
                v_expected = normalize(joints_offsets[child_joint]) #where the child joint is expected to be in T pose
                v_current = keypoints[child_joint] - keypoints[joint]
                v_current = R_parent.inv().apply(normalize(v_current))

                #direction this axis is allowed to rotate around.
                if joint in ['left_knee', 'right_knee']:
                    rotation_axis = np.array([0,1,0])
                else: #elbows
                    rotation_axis = np.array([0,0,1])

                local_joint_rots = []

                for i,v in enumerate(v_current): #iterate over each frame
                    # full rotation matrix
                    R_i, _ = Rotation.align_vectors([v], [v_expected])
                    r_full = R_i.as_rotvec()

                    #project the rotation vector onto 1D rotation axis. This forces only a single direction rotation.
                    r_proj_mag = np.dot(r_full, rotation_axis)
                    r_1D = r_proj_mag * rotation_axis

                    #convert back to quaternion
                    R_quat = Rotation.from_rotvec(r_1D).as_quat()

                    # NOTE: Do NOT do sign-flipping here. 
                    # The rotation vector projection method ensures continuity 
                    # in this single-axis rotation and is typically followed by 
                    # the overall rot_vec smoothing for all joints.
                    local_joint_rots.append(R_quat)
            
            #smooth the quaternion of rotations
            local_joint_rots = utils.smooth_quaternion_rotations_rotvec(local_joint_rots)
            joint_rotations[joint] = local_joint_rots
                        
            plt.plot(local_joint_rots[:, 0], label = 'x')
            plt.plot(local_joint_rots[:, 1], label = 'y')
            plt.plot(local_joint_rots[:, 2], label = 'z')
            plt.plot(local_joint_rots[:, 3], label = 'w')

            plt.ylim(-1.1, 1.1)
            plt.legend()
            plt.title(joint)
            plt.show()

            euler_angles = np.array([Rotation.from_quat(q).as_euler('yxz') for q in local_joint_rots])
            plt.plot(euler_angles[:, 0], label = 'x')
            plt.plot(euler_angles[:, 1], label = 'y')
            plt.plot(euler_angles[:, 2], label = 'z')

            plt.ylim(-3.2, 3.2)
            plt.legend()
            plt.title(joint)
            plt.show()
            #quit()

    return joint_rotations

def main():

    #open the keypoints data
    kpts = np.loadtxt(sys.argv[1])
    kpts = to_dictionary(kpts)
    kpts = utils.smooth_keypoints(kpts, 3) #applies median filter to try and get rid of bad keypoint estimations

    #add the hips and the spine as the midpoint between the waists and the shoulders.
    kpts = add_hip_spine(kpts)
    
    #convert the keypoints to be in root frame.
    kpts_root, root_pos, R_root = to_root_frame(kpts)

    #before we calculate the joint angles, lets get a list of children for each joint
    #returns a dict of joint and their direct children
    children = utils.get_children(joints_heirarchy)

    utils.animate_skeleton(kpts_root, joints_heirarchy)

    #calculate the joint angles. Returns a dict with joints as keys and values with shape: [num_frames, 4].
    #In other words, a quaternion is returned for each joint and each frame
    joint_angles = calculate_joint_angles(kpts_root, joints_heirarchy, joints_offsets, children)
    joint_angles['root_positions'] = root_pos
    joint_angles['root_rotations'] = R_root

    with open("mocap_data.pkl", "wb") as f:
        pickle.dump(joint_angles, f)

main()