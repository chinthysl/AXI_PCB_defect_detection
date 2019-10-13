import cv2
import os
import logging


class SolderJoint:
    def __init__(self, component_name, defect_id, defect_name, roi):
        self.component_name = component_name
        self.defect_id = defect_id
        self.defect_name = defect_name
        self.roi = roi
        self.x_min, self.y_min, self.x_max, self.y_max = roi[0], roi[1], roi[2], roi[3]
        if self.x_max - self.x_min != self.y_max - self.y_min:
            self.is_square = False
        else:
            self.is_square = True

        self.slice_dict = {}

        logging.info('Created solder join, component_name:%s defect_id:%d defect_name:%s roi:%d,%d,%d,%d',
                     self.component_name, self.defect_id, self.defect_name, self.roi[0], self.roi[1], self.roi[2],
                     self.roi[3])

    # this method concat 0,1,2,3 slices in 2d plain if those slices exist and roi is square
    # if you want to concat slices in different method write a new function like this
    def concat_first_four_slices_and_resize(self, width, height):
        logging.debug('start concatenating image, joint type: %s', self.defect_name)
        if not self.is_square:
            logging.debug('joint roi is rectangular, concatenation canceled')
            return None, None

        #  check whether 1st 4 slices are available
        if 0 in self.slice_dict.keys() and 1 in self.slice_dict.keys() and 2 in self.slice_dict.keys() and 3 in self.slice_dict.keys():
            slice_0 = cv2.imread(self.slice_dict[0])
            slice_1 = cv2.imread(self.slice_dict[1])
            slice_2 = cv2.imread(self.slice_dict[2])
            slice_3 = cv2.imread(self.slice_dict[3])

            slice_0_roi = slice_0[self.y_min:self.y_max, self.x_min:self.x_max]
            slice_1_roi = slice_1[self.y_min:self.y_max, self.x_min:self.x_max]
            slice_2_roi = slice_2[self.y_min:self.y_max, self.x_min:self.x_max]
            slice_3_roi = slice_3[self.y_min:self.y_max, self.x_min:self.x_max]

            im_h1 = cv2.hconcat([slice_0_roi, slice_1_roi])
            im_h2 = cv2.hconcat([slice_2_roi, slice_3_roi])

            im_concat = cv2.vconcat([im_h1, im_h2])
            if im_concat is None:
                logging.error('im_concat is none, skipping concatenation')
                return None, None

            resized_image = cv2.resize(im_concat, (width, height), interpolation=cv2.INTER_AREA)
            logging.debug('First 4 slices available, concatenation done')
            return resized_image, self.defect_name

        else:
            logging.error('First 4 slices not available, canceling concatenation')
            return None, None

    def add_slice(self, slice_id, file_location):
        self.slice_dict[slice_id] = file_location
        logging.info('slice id: %d added to the joint, image name: %s', slice_id, file_location)
