import logging
import pickle
import os
import csv
import re
import cv2
import sys

from constants import DEFECT_NAMES_DICT
from utils_basic import chk_n_mkdir

from board_view import BoardView


class SolderJointContainer:
    def __init__(self):
        self.board_view_dict = {}
        self.new_image_name_mapping_dict = {}
        self.csv_details_dict = {}
        self.incorrect_board_view_ids = []

        with open('original_dataset/PTH2_reviewed.csv', mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)

            for row in csv_reader:
                component_name = row["component"]
                defect_id = int(row["defect_type_id"])
                defect_name = DEFECT_NAMES_DICT[defect_id]
                roi = [int(float(str)) for str in row["roi_original"].split()]
                file_location = 'original_dataset\\' + row["image_filename"].strip('C:\Projects\pia-test\\')
                view_identifier = file_location[:-5]

                if view_identifier in self.board_view_dict.keys():
                    logging.debug('fount view_identifier inside the board_view_dict')
                    board_view_obj = self.board_view_dict[view_identifier]
                    board_view_obj.add_slice(file_location)
                else:
                    logging.debug('adding new BoardView obj to the board_view_dict')
                    board_view_obj = BoardView(view_identifier)
                    self.board_view_dict[view_identifier] = board_view_obj

                board_view_obj.add_solder_joint(component_name, defect_id, defect_name, roi)

                # csv_details_dict is only made for file tracking purpose
                if file_location in self.csv_details_dict.keys():
                    self.csv_details_dict[file_location].append([component_name, defect_id, defect_name, roi])
                else:
                    self.csv_details_dict[file_location] = []
                    self.csv_details_dict[file_location].append([component_name, defect_id, defect_name, roi])

                logging.debug('csv row details, component_name:%s, defect_name:%s, roi:%d,%d,%d,%d', component_name,
                              defect_name, roi[0], roi[1], roi[2], roi[3])

            for idx, file_loc in enumerate(self.csv_details_dict.keys()):
                raw_image_name = file_loc[-12:]
                image_name_with_idx = str(idx) + "_" + raw_image_name
                self.new_image_name_mapping_dict[image_name_with_idx] = file_loc

        logging.info('SolderJointContainer obj created')

    # this method create images in a seperate directory marked with rois
    def mark_all_images_with_rois(self):
        logging.info("Num of images to be marked:%d", len(self.csv_details_dict.keys()))
        for idx, file_loc in enumerate(self.csv_details_dict.keys()):
            raw_image_name = file_loc[-12:]
            destination_path = './images_roi_marked/'
            chk_n_mkdir(destination_path)
            destination_image_path = destination_path + str(idx) + "_" + raw_image_name

            src_image = cv2.imread(file_loc)
            if src_image is None:
                logging.error('Could not open or find the image: %s', file_loc)
                exit(0)

            for feature_list in self.csv_details_dict[file_loc]:
                component_name = feature_list[0]
                defect_name = feature_list[2]
                roi = feature_list[3]
                if defect_name == "missing":
                    num = '-mis'
                elif defect_name == "short":
                    num = '-sht'
                else:
                    num = '-inf'

                # draw the ROI
                cv2.rectangle(src_image, (roi[0], roi[1]), (roi[2], roi[3]), (255, 0, 0), 2)
                cv2.putText(src_image, component_name + num, (roi[0], roi[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 0, 255),
                            lineType=cv2.LINE_AA)

            # cv2.imshow("Labeled Image", src_image)
            # k = cv2.waitKey(5000)
            # if k == 27:  # If escape was pressed exit
            #     cv2.destroyAllWindows()
            #     break
            cv2.imwrite(destination_image_path, src_image)
            logging.debug('ROI marked image saved: %s', destination_image_path)

    # this method generates new SolderJoint objs with non_defective labels from xml data
    def load_non_defective_rois_to_container(self):

        annotation_dir = './non_defective_xml_files'
        annotation_files = os.listdir(annotation_dir)

        for file in annotation_files:
            fp = open(annotation_dir + '/' + file, 'r')
            annotation_file = fp.read()
            file_name = annotation_file[annotation_file.index('<filename>') + 10:annotation_file.index('</filename>')]
            x_min_start = [m.start() for m in re.finditer('<xmin>', annotation_file)]
            x_min_end = [m.start() for m in re.finditer('</xmin>', annotation_file)]
            y_min_start = [m.start() for m in re.finditer('<ymin>', annotation_file)]
            y_min_end = [m.start() for m in re.finditer('</ymin>', annotation_file)]
            x_max_start = [m.start() for m in re.finditer('<xmax>', annotation_file)]
            x_max_end = [m.start() for m in re.finditer('</xmax>', annotation_file)]
            y_max_start = [m.start() for m in re.finditer('<ymax>', annotation_file)]
            y_max_end = [m.start() for m in re.finditer('</ymax>', annotation_file)]

            x_min = [int(annotation_file[x_min_start[j] + 6:x_min_end[j]]) for j in range(len(x_min_start))]
            y_min = [int(annotation_file[y_min_start[j] + 6:y_min_end[j]]) for j in range(len(y_min_start))]
            x_max = [int(annotation_file[x_max_start[j] + 6:x_max_end[j]]) for j in range(len(x_max_start))]
            y_max = [int(annotation_file[y_max_start[j] + 6:y_max_end[j]]) for j in range(len(y_max_start))]
            act_file_name = self.new_image_name_mapping_dict[file_name]
            view_identifier = act_file_name[:-5]
            board_view_obj = self.board_view_dict[view_identifier]

            for i in range(len(x_min)):

                x_min_i = x_min[i]
                y_min_i = y_min[i]
                x_max_i = x_max[i]
                y_max_i = y_max[i]
                width = x_max[i] - x_min[i]
                height = y_max[i] - y_min[i]

                logging.debug('Start height/width:' + str(height) + '/' + str(width))
                if width > height:
                    threshold = (width - height) / width * 100.0
                    if threshold > 10.0:
                        logging.debug('height < width:' + str(height) + '/' + str(width))
                        continue
                    else:
                        if (width - height) % 2 == 0:
                            y_min_i = y_min_i - (width - height) // 2
                            y_max_i = y_max_i + (width - height) // 2
                        else:
                            y_min_i = y_min_i - (width - height) // 2 - 1
                            y_max_i = y_max_i + (width - height) // 2

                        height = y_max_i - y_min_i
                        logging.debug('new height/width:' + str(height) + '/' + str(width))

                if width < height:
                    threshold = (height - width) / height * 100.0
                    if threshold > 10.0:
                        logging.debug('height > width:' + str(height) + '/' + str(width))
                        continue
                    else:
                        if (height - width) % 2 == 0:
                            x_min_i = x_min_i - (height - width) // 2
                            x_max_i = x_max_i + (height - width) // 2
                        else:
                            x_min_i = x_min_i - (height - width) // 2 - 1
                            x_max_i = x_max_i + (height - width) // 2

                        width = x_max_i - x_min_i
                        logging.debug('new height/width:' + str(height) + '/' + str(width))

                if not (x_max_i - x_min_i) == (y_max_i - y_min_i):
                    logging.error('w,h,xmin,xmax,ymin,ymax: %d,%d,%d,%d,%d,%d', width, height, x_min_i, x_max_i, y_min_i, y_max_i)
                    sys.exit()

                board_view_obj.add_solder_joint('unknown', -1, 'normal', [x_min_i, y_min_i, x_max_i, y_max_i])
                board_view_obj.add_slices_to_solder_joints()

    @staticmethod
    def create_incorrect_roi_pickle_file(self):
        images_list = os.listdir('./incorrect_roi_images')
        with open('incorrect_roi_images.p', 'wb') as filehandle:
            pickle.dump(images_list, filehandle)

    def find_flag_incorrect_roi_board_view_objs(self):
        with open('incorrect_roi_images.p', 'rb') as filehandle:
            incorrect_roi_list = pickle.load(filehandle)
        temp_list = []
        for image_name in incorrect_roi_list:
            temp_list.append(self.new_image_name_mapping_dict[image_name][:-5])

        list_set = set(temp_list)
        temp_list = list(list_set)
        self.incorrect_board_view_ids.extend(temp_list)

        for board_view_obj in self.board_view_dict.values():
            if board_view_obj.view_identifier in self.incorrect_board_view_ids:
                board_view_obj.is_incorrect_view = True
                logging.debug('marked board view %s as incorrect', board_view_obj.view_identifier)

    def save_concat_images_first_four_slices(self, width=128, height=128):
        chk_n_mkdir('./roi_concatenated_four_slices/short/')
        chk_n_mkdir('./roi_concatenated_four_slices/insufficient/')
        chk_n_mkdir('./roi_concatenated_four_slices/missing/')
        chk_n_mkdir('./roi_concatenated_four_slices/normal/')
        img_count = 0
        for board_view_obj in self.board_view_dict.values():
            if not board_view_obj.is_incorrect_view:
                logging.debug('Concatenating images in board_view_obj: %s', board_view_obj.view_identifier)
                for solder_joint_obj in board_view_obj.solder_joint_dict.values():
                    concat_image, label = solder_joint_obj.concat_first_four_slices_and_resize(width, height)
                    if concat_image is not None:
                        img_count += 1
                        destination_image_path = 'roi_concatenated_four_slices/' + label + '/' + str(img_count) + '.jpg'
                        cv2.imwrite(destination_image_path, concat_image)
                        logging.debug('saving concatenated image, joint type: %s', label)
        logging.info('saved images of concatenated 4 slices in 2d')

    def save_concat_images_all_slices_2d(self):
        chk_n_mkdir('./roi_concatenated_all_slices_2d/short/')
        chk_n_mkdir('./roi_concatenated_all_slices_2d/insufficient/')
        chk_n_mkdir('./roi_concatenated_all_slices_2d/missing/')
        chk_n_mkdir('./roi_concatenated_all_slices_2d/normal/')
        img_count = 0
        for board_view_obj in self.board_view_dict.values():
            if not board_view_obj.is_incorrect_view:
                logging.debug('Concatenating images in board_view_obj: %s', board_view_obj.view_identifier)
                for solder_joint_obj in board_view_obj.solder_joint_dict.values():
                    concat_image, label = solder_joint_obj.concat_pad_all_slices_2d()
                    if concat_image is not None:
                        img_count += 1
                        destination_image_path = 'roi_concatenated_all_slices_2d/' + label + '/' + str(img_count) + '.jpg'
                        cv2.imwrite(destination_image_path, concat_image)
                        logging.debug('saving concatenated image, joint type: %s', label)
        logging.info('saved images of concatenated 6 slices in 2d')

    def print_container_details(self):
        board_views = 0
        solder_joints = 0
        missing_defects = 0
        short_defects = 0
        insuf_defects = 0
        normal_defects = 0

        joints_with_1_slices = 0
        joints_with_2_slices = 0
        joints_with_3_slices = 0
        joints_with_4_slices = 0
        joints_with_5_slices = 0
        joints_with_6_slices = 0

        for board_view_obj in self.board_view_dict.values():
            if not board_view_obj.is_incorrect_view:
                board_views += 1
                for solder_joint_obj in board_view_obj.solder_joint_dict.values():
                    solder_joints += 1

                    if len(solder_joint_obj.slice_dict.keys()) == 1:
                        joints_with_1_slices += 1
                    if len(solder_joint_obj.slice_dict.keys()) == 2:
                        joints_with_2_slices += 1
                    if len(solder_joint_obj.slice_dict.keys()) == 3:
                        joints_with_3_slices += 1
                    if len(solder_joint_obj.slice_dict.keys()) == 4:
                        joints_with_4_slices += 1
                    if len(solder_joint_obj.slice_dict.keys()) == 5:
                        joints_with_5_slices += 1
                    if len(solder_joint_obj.slice_dict.keys()) == 6:
                        joints_with_6_slices += 1

                    label = solder_joint_obj.defect_name
                    if label == 'normal':
                        normal_defects += 1
                    if label == 'missing':
                        missing_defects += 1
                    if label == 'insufficient':
                        insuf_defects += 1
                    if label == 'short':
                        short_defects += 1

        print('*****correct view details*****')
        print('board_views:', board_views, 'solder_joints:', solder_joints, 'missing_defects:', missing_defects,
              'short_defects:', short_defects, 'insuf_defects:', insuf_defects, 'normal_defects:', normal_defects)
        print('joints_with_1_slices:', joints_with_1_slices, 'joints_with_2_slices:', joints_with_1_slices,
              'joints_with_3_slices:', joints_with_3_slices, 'joints_with_4_slices:', joints_with_4_slices,
              'joints_with_5_slices', joints_with_5_slices, 'joints_with_6_slices', joints_with_6_slices)

        board_views = 0
        solder_joints = 0
        missing_defects = 0
        short_defects = 0
        insuf_defects = 0
        normal_defects = 0

        joints_with_1_slices = 0
        joints_with_2_slices = 0
        joints_with_3_slices = 0
        joints_with_4_slices = 0
        joints_with_5_slices = 0
        joints_with_6_slices = 0

        for board_view_obj in self.board_view_dict.values():
            if board_view_obj.is_incorrect_view:
                board_views += 1

                for solder_joint_obj in board_view_obj.solder_joint_dict.values():
                    solder_joints += 1

                    if len(solder_joint_obj.slice_dict.keys()) == 1:
                        joints_with_1_slices += 1
                    if len(solder_joint_obj.slice_dict.keys()) == 2:
                        joints_with_2_slices += 1
                    if len(solder_joint_obj.slice_dict.keys()) == 3:
                        joints_with_3_slices += 1
                    if len(solder_joint_obj.slice_dict.keys()) == 4:
                        joints_with_4_slices += 1
                    if len(solder_joint_obj.slice_dict.keys()) == 5:
                        joints_with_5_slices += 1
                    if len(solder_joint_obj.slice_dict.keys()) == 6:
                        joints_with_6_slices += 1

                    label = solder_joint_obj.defect_name
                    if label == 'normal':
                        normal_defects += 1
                    if label == 'missing':
                        missing_defects += 1
                    if label == 'insufficient':
                        insuf_defects += 1
                    if label == 'short':
                        short_defects += 1

        print('*****incorrect view details*****')
        print('board_views:', board_views, 'solder_joints:', solder_joints, 'missing_defects:', missing_defects,
              'short_defects:', short_defects, 'insuf_defects:', insuf_defects, 'normal_defects:', normal_defects)
        print('joints_with_1_slices:', joints_with_1_slices, 'joints_with_2_slices:', joints_with_1_slices,
              'joints_with_3_slices:', joints_with_3_slices, 'joints_with_4_slices:', joints_with_4_slices,
              'joints_with_5_slices', joints_with_5_slices, 'joints_with_6_slices', joints_with_6_slices)

        board_views = 0
        solder_joints = 0
        missing_defects = 0
        short_defects = 0
        insuf_defects = 0
        normal_defects = 0

        joints_with_1_slices = 0
        joints_with_2_slices = 0
        joints_with_3_slices = 0
        joints_with_4_slices = 0
        joints_with_5_slices = 0
        joints_with_6_slices = 0

        for board_view_obj in self.board_view_dict.values():
            if not board_view_obj.is_incorrect_view:
                board_views += 1
                for solder_joint_obj in board_view_obj.solder_joint_dict.values():
                    if solder_joint_obj.is_square:
                        solder_joints += 1

                        if len(solder_joint_obj.slice_dict.keys()) == 1:
                            joints_with_1_slices += 1
                        if len(solder_joint_obj.slice_dict.keys()) == 2:
                            joints_with_2_slices += 1
                        if len(solder_joint_obj.slice_dict.keys()) == 3:
                            joints_with_3_slices += 1
                        if len(solder_joint_obj.slice_dict.keys()) == 4:
                            joints_with_4_slices += 1
                        if len(solder_joint_obj.slice_dict.keys()) == 5:
                            joints_with_5_slices += 1
                        if len(solder_joint_obj.slice_dict.keys()) == 6:
                            joints_with_6_slices += 1

                        label = solder_joint_obj.defect_name
                        if label == 'normal':
                            normal_defects += 1
                        if label == 'missing':
                            missing_defects += 1
                        if label == 'insufficient':
                            insuf_defects += 1
                        if label == 'short':
                            short_defects += 1

        print('*****correct square roi details*****')
        print('board_views:', board_views, 'solder_joints:', solder_joints, 'missing_defects:', missing_defects,
              'short_defects:', short_defects, 'insuf_defects:', insuf_defects, 'normal_defects:', normal_defects)
        print('joints_with_1_slices:', joints_with_1_slices, 'joints_with_2_slices:', joints_with_1_slices,
              'joints_with_3_slices:', joints_with_3_slices, 'joints_with_4_slices:', joints_with_4_slices,
              'joints_with_5_slices', joints_with_5_slices, 'joints_with_6_slices', joints_with_6_slices)
