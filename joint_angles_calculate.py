import sys
import numpy as np
from scipy.spatial.transform import Rotation
import utils

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


def to_root_frame(kpts):
    """
    Convert mocap keypoints to the root (hip) frame.
    """

    root_pos = kpts["hip"]
    z_axis = kpts["spine"] - kpts["hip"]        #up
    y_axis = kpts["left_waist"] - kpts["hip"]   #left
    x_axis = np.cross(y_axis, z_axis)           #forward

    # Normalize and orthogonalize
    def normalize(v): return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8, None)
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

def calculate_joint_angles(keypoints, joints_heirarchy, joints_offsets, children):
    
    #the longest joints chain in the data
    max_depth = np.max([len(joints) for joints in joints_heirarchy.values()])

    #calculate the joint angles
    joint_rotations = {}
    for depth in range(1, max_depth): #skip root depth
        
        #calculate only at current depth
        for joint, parents in joints_heirarchy.items():
            if len(parents) != depth: continue
                   
            #skip endpoints
            if len(children[joint]) == 0: continue

            if parents[0] == 'hip':
                pass #get the identity rotation
            else:
                #get the rotation of the parent. Shoud this be a chain?
                pass


        pass

def main():

    #open the keypoints data
    kpts = np.loadtxt(sys.argv[1])
    kpts = to_dictionary(kpts)
    kpts = utils.smooth_keypoints(kpts, 3) #applies median filter to try and get rid of bad keypoint estimations

    #add the hips and the spine as the midpoint between the waists and the shoulders.
    kpts = add_hip_spine(kpts)

    #calculate bone lengths from data. Useful for visualizing later.
    bone_lengths = utils.compute_bone_lengths(kpts, joints_heirarchy)
    
    #convert the keypoints to be in root frame.
    kpts_root, root_pos, R_root = to_root_frame(kpts)

    #visualize the keypoints in root frame.
    #utils.animate_skeleton(kpts_root, joints_heirarchy)

    #before we calculate the joint angles, lets get a list of children for each joint
    #returns a dict of joint and their direct children
    children = utils.get_children(joints_heirarchy)

    #calculate the joint angles
    calculate_joint_angles(kpts_root, joints_heirarchy, joints_offsets, children)



main()