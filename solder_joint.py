import cv2
import numpy as np
import logging


class SolderJoint:
    def __init__(self, component_name, defect_id, defect_name, roi):
        self.component_name = component_name
        self.defect_id = defect_id
        self.defect_name = defect_name
        self.roi = roi
        self.x_min, self.y_min, self.x_max, self.y_max = roi[0], roi[1], roi[2], roi[3]
        self.slice_dict = {}

        logging.info('Created solder join, component_name:%s defect_id:%d defect_name:%s roi:%d,%d,%d,%d',
                     self.component_name, self.defect_id, self.defect_name, self.roi[0], self.roi[1], self.roi[2],
                     self.roi[3])

    def is_square(self):
        if (self.x_max - self.x_min) == (self.y_max - self.y_min):
            return True
        else:
            return False

    def add_slice(self, slice_id, file_location):
        self.slice_dict[slice_id] = file_location
        logging.info('slice id: %d added to the joint, image name: %s', slice_id, file_location)

    # this method concat 0,1,2,3 slices in 2d plain if those slices exist and roi is square
    # if you want to concat slices in different method write a new function like this
    def concat_first_four_slices_and_resize(self, width, height):
        logging.debug('start concatenating image, joint type: %s', self.defect_name)

        if not self.is_square():
            logging.debug('joint roi is rectangular, canceling concatenation')
            return None, None

        if len(self.slice_dict.keys()) < 4:
            logging.error('Number of slice < 4, canceling concatenation')
            return None, None

        if len(self.slice_dict.keys()) < 4:
            logging.error('Number of slice > 4, canceling concatenation')
            return None, None

        #  check whether 1st 4 slices are available
        if 0 in self.slice_dict.keys() and 1 in self.slice_dict.keys() and 2 in self.slice_dict.keys() and 3 in self.slice_dict.keys():
            slice_0 = cv2.imread(self.slice_dict[0])
            slice_1 = cv2.imread(self.slice_dict[1])
            slice_2 = cv2.imread(self.slice_dict[2])
            slice_3 = cv2.imread(self.slice_dict[3])

            slice_0 = cv2.cvtColor(slice_0, cv2.COLOR_BGR2GRAY)
            slice_1 = cv2.cvtColor(slice_1, cv2.COLOR_BGR2GRAY)
            slice_2 = cv2.cvtColor(slice_2, cv2.COLOR_BGR2GRAY)
            slice_3 = cv2.cvtColor(slice_3, cv2.COLOR_BGR2GRAY)

            slice_0_roi = slice_0[self.y_min:self.y_max, self.x_min:self.x_max]
            slice_1_roi = slice_1[self.y_min:self.y_max, self.x_min:self.x_max]
            slice_2_roi = slice_2[self.y_min:self.y_max, self.x_min:self.x_max]
            slice_3_roi = slice_3[self.y_min:self.y_max, self.x_min:self.x_max]

            im_h1 = cv2.hconcat([slice_0_roi, slice_1_roi])
            im_h2 = cv2.hconcat([slice_2_roi, slice_3_roi])

            im_concat = cv2.vconcat([im_h1, im_h2])
            if im_concat is None:
                logging.error('Error occured in opencv ROI concat, is none, skipping concatenation')
                return None, None

            resized_image = cv2.resize(im_concat, (width, height), interpolation=cv2.INTER_AREA)
            logging.debug('First 4 slices available, concatenation done')
            return resized_image, self.defect_name

        else:
            logging.error('First 4 slices not available, canceling concatenation')
            return None, None

    def concat_pad_all_slices_2d(self):
        logging.debug('start concatenating image, joint type: %s', self.defect_name)

        if not self.is_square():
            logging.error('joint roi is rectangular, canceling concatenation')
            return None, None

        blank_image = np.zeros(shape=[self.x_max - self.x_min, self.y_max - self.y_min], dtype=np.uint8)
        slices_list = [None, None, None, None, None, None]
        for slice_id in range(6):
            if slice_id in self.slice_dict.keys():
                img = cv2.imread(self.slice_dict[slice_id])
                # there's a bug here. image slicing doesn't give a perfect square sometimes
                img_roi = img[self.y_min:self.y_max, self.x_min:self.x_max]
                img_roi_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)

                if img_roi_gray is None:
                    logging.error('Error occured in opencv ROI extraction')
                    return None, None

                if blank_image.shape != img_roi_gray.shape:
                    logging.error('Error occured in opencv ROI extraction shape')
                    return None, None
                slices_list[slice_id] = img_roi_gray
            else:
                slices_list[slice_id] = blank_image
                logging.debug('blank slice added to slice: %d', slice_id)

        # logging.debug('xmax, xmin, ymax, ymin: %d, %d, %d, %d', self.x_max, self.x_min, self.y_max, self.y_min)
        # logging.debug('Slices shapes: %s, %s, %s, %s, %s, %s', slices_list[0].shape, slices_list[1].shape,
        #               slices_list[2].shape, slices_list[3].shape, slices_list[4].shape, slices_list[5].shape)

        im_h1 = cv2.hconcat(slices_list[0:3])
        im_h2 = cv2.hconcat(slices_list[3:6])
        im_concat = cv2.vconcat([im_h1, im_h2])

        if im_concat is None:
            logging.error('im_concat is none, skipping concatenation')
            return None, None

        resized_image = cv2.resize(im_concat, (192, 128), interpolation=cv2.INTER_AREA)
        logging.debug('concatenation done')
        return resized_image, self.defect_name
