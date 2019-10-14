# AXI_PCB_defect_detection

This repo contains data pre-processing, classification and defect detection methodologies for images from **Advance XRay Inspection** from multi-layer PCB boards.

- [AXI_PCB_defect_detection](#AXI_PCB_defect_detection)
  * [General guidelines for contributors](#general-guidelines)
  * [Project file structure](#project-file-structure)
  * [Solution Overview](#solution-overview)
  * [Code explanation](#code-explanation)
    + [solder_joint.py](#solder_joint)
    + [board_view.py](#board_view)
    + [solder_joint_container.py](#solder_joint_container)
    + [main.py](#main)
  * [DVC integration](#dvc-integration)

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

This section will explains about setting up the data version controlling and storage for the project. Currently we don't do DVC.
