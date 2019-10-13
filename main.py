import logging
import pickle

from solder_joint_container import SolderJointContainer

logging.basicConfig(level=logging.DEBUG)

# if you don't have the pickle file for the container obj make CREATE_OBJ = True
CREATE_OBJ = False
if CREATE_OBJ:
    # creating solder joint object will read the data set and create all other objects
    solder_joint_container_obj = SolderJointContainer()
    # from a single slice of incorrect roi images whole board view objects will be marked
    solder_joint_container_obj.find_flag_incorrect_roi_board_view_objs()
    # load non defective solder joint info to the container
    solder_joint_container_obj.load_non_defective_rois_to_container()

    with open('solder_joint_container_obj.p', 'wb') as output_handle:
        pickle.dump(solder_joint_container_obj, output_handle, pickle.HIGHEST_PROTOCOL)

# **********************************************************************************************************************
with open('solder_joint_container_obj.p', 'rb') as input_handle:
    solder_joint_container_obj = pickle.load(input_handle)

# print details of BoardView objects and SolderJoint objects
solder_joint_container_obj.print_container_details()

# this method will save concat 2d cleaned images in a directory
solder_joint_container_obj.save_concat_images_first_four_slices()

# # this method will create a directory full of images with marked rois
# solder_joint_container_obj.mark_all_images_with_rois()
