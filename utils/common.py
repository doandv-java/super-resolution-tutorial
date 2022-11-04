import os

import json
import numpy as np
import math
import shutil


def load_file_json(json_file):
    if os.path.exists(json_file):
        file = open(json_file)
        return json.load(file)
    else:
        print("Not found file json config in:{}".format(json_file))
        return None


def is_image_file(file_path: str):
    return any(file_path.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.JPG'])


def clear_folder(folder_dir):
    for filename in os.listdir(folder_dir):
        file_path = os.path.join(folder_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def get_img_from_directory(img_list: list, root: str):
    if os.path.isdir(root):
        for sub_file in os.listdir(root):
            sub_path = os.path.join(root, sub_file)
            get_img_from_directory(img_list, sub_path)
    elif is_image_file(root):
        img_list.append(root)


# Get shape of patches when you divide image by patch size
def patch_dims(mat_size, patch_size):
    return np.ceil(np.array(mat_size) / patch_size).astype(int)


# Create list patch image by divide image by patch size
def create_patches(mat, patch_size):
    mat_size = mat.shape
    patches_dim = patch_dims(mat_size=mat_size[:2], patch_size=patch_size)
    patches_count = np.product(patches_dim)

    patches = np.zeros(shape=(patches_count, patch_size, patch_size, 3), dtype=np.float32)
    for y in range(patches_dim[0]):
        y_start = y * patch_size
        for x in range(patches_dim[1]):
            x_start = x * patch_size
            single_patch = mat[y_start: y_start + patch_size, x_start: x_start + patch_size, :]
            # zero pad patch in bottom and right side if real patch size is smaller than patch size
            real_patch_h, real_patch_w = single_patch.shape[:2]
            patch_id = y + x * patches_dim[0]
            patches[patch_id, :real_patch_h, :real_patch_w, :] = single_patch

    return patches


# Assemble patch image from list patch image
def assemble_patches(patches, mat_size, patch_size, up_size=1):
    patch_dim_h, patch_dim_w = patch_dims(mat_size=mat_size[:2], patch_size=patch_size)
    assemble_image = np.zeros(shape=(patch_size * patch_dim_h * up_size, patch_size * patch_dim_w * up_size, 3),
                              dtype=np.uint8)
    patches_count = np.product((patch_dim_h, patch_dim_w))

    for i in range(patches_count):
        y = (i % patch_dim_h) * patch_size * up_size
        x = int(math.floor(i / patch_dim_h)) * patch_size * up_size

        assemble_image[y:y + patch_size * up_size, x:x + patch_size * up_size, :] = patches[i]
    assemble_image = assemble_image[:mat_size[0]*up_size, :mat_size[1]*up_size, :]
    return assemble_image
