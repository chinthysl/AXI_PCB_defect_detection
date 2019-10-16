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
    def concat_first_four_slices_2d(self):
        logging.debug('start concatenating image, joint type: %s', self.defect_name)

        if not self.is_square():
            logging.debug('joint roi is rectangular, canceling concatenation')
            return None, None

        if len(self.slice_dict.keys()) < 4:
            logging.error('Number of slice < 4, canceling concatenation')
            return None, None

        if len(self.slice_dict.keys()) > 4:
            logging.error('Number of slice > 4, canceling concatenation')
            return None, None

        slices_list = [None, None, None, None]
        for slice_id in range(4):
            #  check whether 1st 4 slices are available
            if slice_id not in self.slice_dict.keys():
                logging.error('First 4 slices not available, canceling concatenation')
                return None, None

            img = cv2.imread(self.slice_dict[slice_id])
            # there's a bug here. image slicing doesn't give a perfect square sometimes
            img_roi = img[self.y_min:self.y_max, self.x_min:self.x_max]
            img_roi_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
            if img_roi_gray is None:
                logging.error('Slice read is None, canceling concatenation')
                return None, None
            resized_image = cv2.resize(img_roi_gray, (128, 128), interpolation=cv2.INTER_AREA)

            if resized_image is None:
                logging.error('Error occured in opencv ROI extraction')
                return None, None

            slices_list[slice_id] = resized_image

        im_h1 = cv2.hconcat(slices_list[0:2])
        im_h2 = cv2.hconcat(slices_list[2:4])
        im_concat = cv2.vconcat([im_h1, im_h2])

        if im_concat is None:
            logging.error('Error occured in opencv ROI concat, is none, skipping concatenation')
            return None, None

        logging.debug('First 4 slices available, concatenation done')
        return im_concat, self.defect_name

    def concat_first_four_slices_3d(self):
        logging.debug('start concatenating image, joint type: %s', self.defect_name)

        if not self.is_square():
            logging.debug('joint roi is rectangular, canceling concatenation')
            return None, None

        if len(self.slice_dict.keys()) < 4:
            logging.error('Number of slice < 4, canceling concatenation')
            return None, None

        if len(self.slice_dict.keys()) > 4:
            logging.error('Number of slice > 4, canceling concatenation')
            return None, None

        logging.debug('start concatenating image, joint type: %s', self.defect_name)

        slices_list = [None, None, None, None]
        for slice_id in range(4):
            #  check whether 1st 4 slices are available
            if slice_id not in self.slice_dict.keys():
                logging.error('First 4 slices not available, canceling concatenation')
                return None, None

            img = cv2.imread(self.slice_dict[slice_id])
            # there's a bug here. image slicing doesn't give a perfect square sometimes
            img_roi = img[self.y_min:self.y_max, self.x_min:self.x_max]
            img_roi_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
            if img_roi_gray is None:
                logging.error('Slice read is None, canceling concatenation')
                return None, None
            resized_image = cv2.resize(img_roi_gray, (128, 128), interpolation=cv2.INTER_AREA)
            resized_image = resized_image.astype(np.float32) / 255

            if resized_image is None:
                logging.error('Error occured in opencv ROI extraction')
                return None, None

            slices_list[slice_id] = resized_image

        # logging.debug(slices_list[0].shape)
        stacked_np_array = np.stack(slices_list, axis=2)
        # logging.debug(stacked_np_array.shape)
        stacked_np_array = np.expand_dims(stacked_np_array, axis=4)
        logging.debug('3d image shape: %s', stacked_np_array.shape)

        logging.debug('First 4 slices available, concatenation done')
        return stacked_np_array, self.defect_name

    def concat_pad_all_slices_2d(self):
        logging.debug('start concatenating image, joint type: %s', self.defect_name)

        if not self.is_square():
            logging.error('joint roi is rectangular, canceling concatenation')
            return None, None

        blank_image = np.zeros(shape=[128, 128], dtype=np.uint8)
        slices_list = [None, None, None, None, None, None]
        for slice_id in range(6):
            if slice_id in self.slice_dict.keys():
                img = cv2.imread(self.slice_dict[slice_id])
                # there's a bug here. image slicing doesn't give a perfect square sometimes
                img_roi = img[self.y_min:self.y_max, self.x_min:self.x_max]
                img_roi_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
                if img_roi_gray is None:
                    logging.error('Slice read is None, canceling concatenation')
                    return None, None
                resized_image = cv2.resize(img_roi_gray, (128, 128), interpolation=cv2.INTER_AREA)

                if resized_image is None:
                    logging.error('Error occured in opencv ROI extraction')
                    return None, None
                slices_list[slice_id] = resized_image

            else:
                slices_list[slice_id] = blank_image
                logging.debug('blank slice added to slice: %d', slice_id)

        im_h1 = cv2.hconcat(slices_list[0:3])
        im_h2 = cv2.hconcat(slices_list[3:6])
        im_concat = cv2.vconcat([im_h1, im_h2])

        if im_concat is None:
            logging.error('im_concat is none, skipping concatenation')
            return None, None

        logging.debug('concatenation done')
        return im_concat, self.defect_name

    def concat_pad_all_slices_3d(self):
        logging.debug('start concatenating image, joint type: %s', self.defect_name)

        if not self.is_square():
            logging.error('joint roi is rectangular, canceling concatenation')
            return None, None

        blank_image = np.zeros(shape=[128, 128], dtype=np.uint8)
        slices_list = [None, None, None, None, None, None]
        for slice_id in range(6):
            if slice_id in self.slice_dict.keys():
                img = cv2.imread(self.slice_dict[slice_id])
                # there's a bug here. image slicing doesn't give a perfect square sometimes
                img_roi = img[self.y_min:self.y_max, self.x_min:self.x_max]
                img_roi_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
                if img_roi_gray is None:
                    logging.error('Slice read is None, canceling concatenation')
                    return None, None
                resized_image = cv2.resize(img_roi_gray, (128, 128), interpolation=cv2.INTER_AREA)
                resized_image = resized_image.astype(np.float32) / 255

                if resized_image is None:
                    logging.error('Error occured in opencv ROI extraction')
                    return None, None

                slices_list[slice_id] = resized_image

            else:
                slices_list[slice_id] = blank_image
                logging.debug('blank slice added to slice: %d', slice_id)

        # logging.debug('xmax, xmin, ymax, ymin: %d, %d, %d, %d', self.x_max, self.x_min, self.y_max, self.y_min)
        # logging.debug('Slices shapes: %s, %s, %s, %s, %s, %s', slices_list[0].shape, slices_list[1].shape,
        #               slices_list[2].shape, slices_list[3].shape, slices_list[4].shape, slices_list[5].shape)

        # logging.debug(slices_list[0].shape)
        stacked_np_array = np.stack(slices_list, axis=2)
        # logging.debug(stacked_np_array.shape)
        stacked_np_array = np.expand_dims(stacked_np_array, axis=4)
        logging.debug('3d image shape: %s', stacked_np_array.shape)

        logging.debug('Padded and concatenated 6 slices in 3d')
        return stacked_np_array, self.defect_name

    def concat_pad_all_slices_inverse_3d(self):
        logging.debug('start concatenating image, joint type: %s', self.defect_name)

        if not self.is_square():
            logging.error('joint roi is rectangular, canceling concatenation')
            return None, None

        blank_image = np.ones(shape=[128, 128], dtype=np.uint8)
        slices_list = [None, None, None, None, None, None]
        for slice_id in range(6):
            if slice_id in self.slice_dict.keys():
                img = cv2.imread(self.slice_dict[slice_id])
                # there's a bug here. image slicing doesn't give a perfect square sometimes
                img_roi = img[self.y_min:self.y_max, self.x_min:self.x_max]
                img_roi_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
                img_roi_gray = cv2.bitwise_not(img_roi_gray)
                if img_roi_gray is None:
                    logging.error('Slice read is None, canceling concatenation')
                    return None, None
                resized_image = cv2.resize(img_roi_gray, (128, 128), interpolation=cv2.INTER_AREA)
                resized_image = resized_image.astype(np.float32) / 255

                if resized_image is None:
                    logging.error('Error occured in opencv ROI extraction')
                    return None, None

                slices_list[slice_id] = resized_image

            else:
                slices_list[slice_id] = blank_image
                logging.debug('blank slice added to slice: %d', slice_id)

        # logging.debug(slices_list[0].shape)
        stacked_np_array = np.stack(slices_list, axis=2)
        # logging.debug(stacked_np_array.shape)
        stacked_np_array = np.expand_dims(stacked_np_array, axis=4)
        logging.debug('3d image shape: %s', stacked_np_array.shape)

        logging.debug('Padded and concatenated 6 slices in 3d')
        return stacked_np_array, self.defect_name
