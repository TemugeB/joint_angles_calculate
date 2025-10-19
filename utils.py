import numpy as np


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

    return bone_lengths