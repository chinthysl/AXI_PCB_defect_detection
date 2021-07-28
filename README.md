# AXI_PCB_defect_detection

This repo contains data pre-processing, classification and defect detection methodologies for images from **Advance XRay Inspection** from multi-layer PCB boards.
Please go through our research paper if you need more details and be kind enough to Cite it if you are going to use this implementation.
- https://ieeexplore.ieee.org/abstract/document/9442142/?casa_token=FNPEvLNODrgAAAAA:Lwm3mFCDg-BgJiSLl1uhefLUv_ApdkNBMbwECzTi1KEGnGX1PohgRLILGQKf3l7Dugr-vuQ7gDZdt4U
- @inproceedings{zhang2020deep,
  title={Deep Learning Based Defect Detection for Solder Joints on Industrial X-Ray Circuit Board Images},
  author={Zhang, Qianru and Zhang, Meng and Gamanayake, Chinthaka and Yuen, Chau and Geng, Zehao and Jayasekaraand, Hirunima and Zhang, Xuewen and Woo, Chia-wei and Low, Jenny and Liu, Xiang},
  booktitle={2020 IEEE 18th International Conference on Industrial Informatics (INDIN)},
  volume={1},
  pages={74--79},
  year={2020},
  organization={IEEE}
}

- [AXI_PCB_defect_detection](#AXI_PCB_defect_detection)
  * [Details of the dataset](#general-guidelines)
  * [General guidelines for contributors](#general-guidelines)
  * [Project file structure](#project-file-structure)
  * [Code explanation](#code-explanation)
    + [solder_joint.py](#solder_joint)
    + [board_view.py](#board_view)
    + [solder_joint_container.py](#solder_joint_container)
    + [main.py](#main)
  * [DVC integration](#dvc-integration)

## Details of the proprietory dataset
- Distinct .jpg images including all slices – 32377
- Distinct ROIs with labels – 92208
- PCB types – 17
- Distinct XRay board views – 8063
- Views containing correct ROIs – 6112, Views containing incorrect ROIs – 1951
- Slices per XRay board view (slices per physical solder joint) – 3, 4, 5, 6
- Number of solder joints – 22872
- Correct joints – 15613,  Incorrect joints – 7259
- Correct-square solder joints – 14672
- missing - 1496, short - 5605, insufficient - 4691 normal - 2880
- 3 sliced joints - 5471, 4 sliced joints - 8590, 5 sliced joints - 97, 6 sliced joints - 277

#### Special Notes:
- A single XRay board view has multiple ROIs marked.
- A single defective ROI(solder joint) can have multiple defect labels.
- If a XRay view contains one incorrect ROI, all solder joints in that view are considered as incorrect.


## General Guidelines for contributors
- Please add your data folders into the ignore file until DVC is setup
- Don't do `git add --all`. Please be aware of what you commit.

## Project file structure

```bash
├── non_defective_xml_files/        # put xml labels for non defective rois here
├── original_dataset/               # put two image folders and PTH2_reviewed.csv inside this
├── board_view.py                   # python class for a unique PCB BoardView
├── constants.py                    # constant values for the project
├── incorrect_roi_images.p          # this file contains file names of all incorrect rois
├── main.py                         # script for creating objects and run ROI concatenation
├── solder_joint.py                 # python class for a unique PCB SolderJoint
├── solder_joint_container.py       # python class containing all BoardView and SolderJoint objs
├── solder_joint_container_obj.p    # saved pickle obj for SolderJointContainer class
└── utils_basics.py                 # helper functions
```

## Code explanation

### solder_joint.py

```
+-----------------------------------------------------------------------+
|                           SolderJoint                                 |
+-----------------------------------------------------------------------+
|+ self.component_name                                                  |
|+ self.defect_id                                                       |
|+ self.defect_name                                                     |
|+ self.roi                                                             |
|+ self.is_square                                                       |
|+ self.slice_dict                                                      |
+-----------------------------------------------------------------------+
|+__init__(self, component_name, defect_id, defect_name, roi)           |
|+add_slice(self, slice_id, file_location)                              |
|+concat_first_four_slices_and_resize(self, width, height)              |
+-----------------------------------------------------------------------+
```

- `add_slice(self, slice_id, file_location)`:

  This method will add the slice id and corresponding image location of that slice to the `self.slice_dict`.

- `concat_first_four_slices_and_resize(self, width, height)`:

  In this method only 1st 4 slices from ids 0,1,2,3 are concatenated in a 2d square shape. Concatenated 2d image and the label of the joint is returned. You have to catch them in `SolderJointContainer` object and save to disk accordingly.

  If you want to write a new method of channel concatenation please modify this method. (For an example, if you want to generate gray scale image per slice and concatenate them as a pickle or numpy file)

### board_view.py

```
+-----------------------------------------------------------------------+
|                           BoardView                                   |
+-----------------------------------------------------------------------+
|+ self.view_identifier                                                 |
|+ self.is_incorrect_view                                               |
|+ self.solder_joint_dict                                               |
|+ self.slice_dict                                                      |
+-----------------------------------------------------------------------+
|+__init__(self, view_identifier)                                       |
|+add_solder_joint(self, component, defect_id, defect_name, roi)        |
|+add_slice(self, file_location)                                        |
|+add_slices_to_solder_joints(self)                                     |
+-----------------------------------------------------------------------+
```

- This class contains all the SolderJoint objects and all the slices details regarding that board view of the PCB.

### solder_joint_container.py

```
+-----------------------------------------------------------------------+
|                           SolderJointContainer                        |
+-----------------------------------------------------------------------+
|+ self.board_view_dict                                                 |
|+ self.new_image_name_mapping_dict                                     |
|+ self.csv_details_dict                                                |
|+ self.incorrect_board_view_ids                                        |
+-----------------------------------------------------------------------+
|+__init__(self)                                                        |
|+mark_all_images_with_rois(self)                                       |
|+load_non_defective_rois_to_container(self)                            |
|+find_flag_incorrect_roi_board_view_objs(self)                         |
|+save_concat_images_first_four_slices(self, width=128, height=128)     |
|+print_container_details(self)                                         |
|+write_csv_defect_roi_images(self)                                     |
+-----------------------------------------------------------------------+
```

- `save_concat_images_first_four_slices(self, width=128, height=128)`:

  In this method only 1st 4 slices from ids 0,1,2,3 are concatenated in a 2d square shape. Concatenated 2d image and the label of the joints returned from the `SolderJoint` object is saved to the disk accordingly.

  If you want to write a new method of channel concatenation please modify this method.

### main.py

`SolderJointContainer ` object is created inside this script. Then using that object we can call the required method to generate concatenated images.

## DVC integration

This section will explains about setting up the data version controlling and storage for the project. Data version control is integrated in this project. You can simple train any model by changing parameters of the neural network models. After training you'll see changes in data folders. DVC commit those changes followed by a git commit. Trained models and data will be tracked by a local DVC storage.
