import logging

from solder_joint import SolderJoint


class BoardView:
    def __init__(self, view_identifier):
        self.view_identifier = view_identifier
        self.solder_joint_dict = {}
        self.slice_dict = {}
        self.is_incorrect_view = False

        logging.info('BoardView obj created for view id: %s', self.view_identifier)

    def add_solder_joint(self, component, defect_id, defect_name, roi):
        my_tupple = tuple([roi[0], roi[1], roi[2], roi[3], defect_id])
        if my_tupple in self.solder_joint_dict.keys():
            logging.info('ROI+Defect found inside the solder_joint_dict, won\'t add a new joint')
        else:
            logging.info('Adding new SolderJoint obj for the new ROI+Defect')
            self.solder_joint_dict[my_tupple] = SolderJoint(component, defect_id, defect_name, roi)

    def add_slice(self, file_location):
        slice_id = int(file_location[-5])
        self.slice_dict[slice_id] = file_location
        for solder_joint_obj in self.solder_joint_dict.values():
            solder_joint_obj.add_slice(slice_id, file_location)

    def add_slices_to_solder_joints(self):
        for slice_id in self.slice_dict.keys():
            file_location = self.slice_dict[slice_id]
            for solder_joint_obj in self.solder_joint_dict.values():
                solder_joint_obj.add_slice(slice_id, file_location)

