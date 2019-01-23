#!/usr/bin/env python3

"""
Copyright 2019 4Tzones

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies
or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import argparse
import os
import yaml
import uuid
import cv2
import random
import imgaug as ia
import numpy as np
from glob import glob
from imgaug import augmenters as iaa
from abc import ABCMeta, abstractmethod
from typing import Tuple


class Dataset:

    __metaclass__ = ABCMeta

    def __init__(self, name):
        self.name = name
        self._images_dir = 'images'
        self._labels_dir = 'labels'

        self.present_label = 0
        self.not_present_label = 1
        self.red_label = 2
        self.yellow_label = 3
        self.green_label = 4

    @abstractmethod
    def get_all_labels(self, input_dir: str) -> list:
        raise NotImplementedError()

    @abstractmethod
    def filter_original_labels(self, labels_file_content: list) -> list:
        raise NotImplementedError()

    @abstractmethod
    def get_class_mapping(self, mode: str):
        raise NotImplementedError()

    @abstractmethod
    def get_output_labels_line(self, entry, bboxes: np.ndarray, output_image_path: str, mode: str) -> str:
        raise NotImplementedError()

    @abstractmethod
    def get_input_image_path(self, input_dir, entry):
        raise NotImplementedError()

    @abstractmethod
    def get_bounding_boxes(self, entry, orig_image_shape):
        raise NotImplementedError()

    @abstractmethod
    def get_light_counters(self, entry) -> Tuple[int, int, int, int]:  # red, yellow, green, no light
        raise NotImplementedError()

    @abstractmethod
    def get_entry_classes(self, entry) -> set:
        raise NotImplementedError()

    def get_output_images_dir(self, output_dir: str) -> str:
        return os.path.join(output_dir, self._images_dir)

    def get_output_image_path(self, output_dir: str, input_file_name: str) -> str:
        img_dir_name = self.get_output_images_dir(output_dir)
        extension = input_file_name.split('.')[-1]
        return os.path.join(img_dir_name, str(uuid.uuid4())) + '.' +  extension

    def get_output_labels_dir(self, output_dir: str) -> str:
        return os.path.join(output_dir, self._labels_dir)

    def get_output_labels_singular_file_name(self, output_dir: str):
        return os.path.join(self.get_output_labels_dir(output_dir), "labels_singular.txt")

    def get_output_labels_ternary_file_name(self, output_dir: str):
        return os.path.join(self.get_output_labels_dir(output_dir), "labels_ternary.txt")

    def get_output_labels_singular_yaml_file_name(self, output_dir: str):
        return os.path.join(self.get_output_labels_dir(output_dir), "labels_singular.yaml")

    def get_output_labels_ternary_vatsal_yaml_file_name(self, output_dir: str):
        return os.path.join(self.get_output_labels_dir(output_dir), "labels_ternary_vatsal.yaml")

    def get_output_labels_ternary_bosh_yaml_file_name(self, output_dir: str):
        return os.path.join(self.get_output_labels_dir(output_dir), "labels_ternary_bosh.yaml")

    def get_output_labels_ternary_yolo_mark_image_file_name(self, image_path: str):
        return image_path.replace(image_path[image_path.rfind('.')+1:], 'txt')

    def get_statistics_file_name(self, output_dir: str):
        return os.path.join(output_dir, "statistics.txt")

    @staticmethod
    def get_present_label():
        return 0

    @staticmethod
    def get_red_label():
        return 0

    @staticmethod
    def get_yellow_label():
        return 1

    @staticmethod
    def get_green_label():
        return 2


class BoschSmallTrafficLightsDataset(Dataset):

    def __init__(self):
        super(BoschSmallTrafficLightsDataset, self).__init__('bosch_small_traffic_lights')
        self._labels = "train.yaml"

        self.label_set = {'GreenLeft', 'RedStraightLeft', 'GreenRight', 'GreenStraightLeft', 'RedStraight',
                          'GreenStraightRight', 'Green', 'GreenStraight', 'RedLeft', 'Yellow', 'RedRight', 'Red'}

        self._singular_class_mapping = {label: self.get_present_label() for label in self.label_set}

        self._ternary_class_mapping = {label: self._choose_label(label) for label in self.label_set}

    @classmethod
    def _choose_label(cls, name):
        if name.startswith('Red'):
            return cls.get_red_label()
        elif name.startswith('Yellow'):
            return cls.get_yellow_label()
        elif name.startswith('Green'):
            return cls.get_green_label()
        else:
            raise ValueError("unknown label name: " + name)

    def get_all_labels(self, input_dir: str) -> list:
        with open(os.path.join(input_dir, self._labels)) as f:
            labels = yaml.safe_load(f)
        return labels

    def filter_original_labels(self, content):
        """
        Chooses only entries with no traffic lights or with non-occluded traffic lights or without off traffic lights.
        :param content: initial content of yaml file
        :return: filtered content of initial yaml file
        """
        filtered_labels = []
        for entry in content:
            exclude = False
            for props in entry['boxes']:
                exclude = exclude or props['occluded'] or props['label'] == 'off'
                if exclude:
                    break
            if not exclude:
                filtered_labels.append(entry)

        return filtered_labels

    def get_class_mapping(self, mode: str):
        if mode == 'singular':
            return self._singular_class_mapping
        elif mode == 'ternary':
            return self._ternary_class_mapping
        else:
            raise ValueError("unknown mode: " + mode)

    def get_output_labels_line(self, entry, bboxes: np.ndarray, output_image_path: str, mode: str) -> str:
        class_mapping = self.get_class_mapping(mode)
        line = output_image_path
        assert bboxes.shape[0] == len(entry['boxes'])
        for i in range(bboxes.shape[0]):
            line += ' ' + str(int(round(bboxes[i][0]))) + ',' \
                        + str(int(round(bboxes[i][1]))) + ',' \
                        + str(int(round(bboxes[i][2]))) + ',' \
                        + str(int(round(bboxes[i][3]))) + ',' \
                        + str(class_mapping[entry['boxes'][i]['label']])
        return line

    def get_input_image_path(self, input_dir, entity):
        return os.path.join(input_dir, entity['path'])

    def get_bounding_boxes(self, entry, orig_image_shape):
        bboxes = []
        for bbox in entry['boxes']:
            bboxes.append([bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max']])
        return np.asarray(bboxes)

    def get_light_counters(self, entry) -> Tuple[int, int, int, int]:
        class_mapping = self.get_class_mapping('ternary')
        counters = [0, 0, 0, 0]
        for bbox in entry['boxes']:
            counters[class_mapping[bbox['label']]] += 1
        if not entry['boxes']:
            counters[3] += 1
        return counters[0], counters[1], counters[2], counters[3]

    def get_entry_classes(self, entry) -> set:
        cls_set = set()
        class_mapping = self.get_class_mapping('ternary')
        for bbox in entry['boxes']:
            cls_set.add(class_mapping[bbox['label']])
        return cls_set


class VatsalSrivastavaTrafficLightsSimulatorDataset(Dataset):

    def __init__(self):
        super(VatsalSrivastavaTrafficLightsSimulatorDataset, self).__init__(
            'vatsal_srivastava_traffic_lights_simulator')
        self._labels = "sim_data_annotations.yaml"

        self.label_set = {'Green', 'Yellow', 'Red'}

        self._singular_class_mapping = {label: self.get_present_label() for label in self.label_set}

        self._ternary_class_mapping = {label: self._choose_label(label) for label in self.label_set}

    @classmethod
    def _choose_label(cls, name):
        if name.startswith('Red'):
            return cls.get_red_label()
        elif name.startswith('Yellow'):
            return cls.get_yellow_label()
        elif name.startswith('Green'):
            return cls.get_green_label()
        else:
            raise ValueError("unknown label name: " + name)

    def get_all_labels(self, input_dir: str) -> list:
        with open(os.path.join(input_dir, self._labels)) as f:
            labels = yaml.safe_load(f)
        return labels

    def filter_original_labels(self, labels_file_content: list) -> list:
        """
        No need to filter anything here.
        :param labels_file_content: original labels content
        :return: filtered labels content
        """
        return labels_file_content

    def get_class_mapping(self, mode: str):
        if mode == 'singular':
            return self._singular_class_mapping
        elif mode == 'ternary':
            return self._ternary_class_mapping
        else:
            raise ValueError("unknown mode: " + mode)

    def get_output_labels_line(self, entry, bboxes: np.ndarray, output_image_path: str, mode: str) -> str:
        class_mapping = self.get_class_mapping(mode)
        line = output_image_path
        assert bboxes.shape[0] == len(entry['annotations'])
        for i in range(bboxes.shape[0]):
            line += ' ' + str(int(round(bboxes[i][0]))) + ',' \
                        + str(int(round(bboxes[i][1]))) + ',' \
                        + str(int(round(bboxes[i][2]))) + ',' \
                        + str(int(round(bboxes[i][3]))) + ',' \
                        + str(class_mapping[entry['annotations'][i]['class']])
        return line

    def get_input_image_path(self, input_dir, entity):
        return os.path.join(input_dir, entity['filename'])

    def get_bounding_boxes(self, entry, orig_image_shape):
        bboxes = []
        for bbox in entry['annotations']:
            bboxes.append([bbox['xmin'], bbox['ymin'], bbox['xmin'] + bbox['x_width'], bbox['ymin'] + bbox['y_height']])
        return np.asarray(bboxes)

    def get_light_counters(self, entry) -> Tuple[int, int, int, int]:
        class_mapping = self.get_class_mapping('ternary')
        counters = [0, 0, 0, 0]
        for bbox in entry['annotations']:
            counters[class_mapping[bbox['class']]] += 1
        if not entry['annotations']:
            counters[3] += 1
        return counters[0], counters[1], counters[2], counters[3]

    def get_entry_classes(self, entry) -> set:
        cls_set = set()
        class_mapping = self.get_class_mapping('ternary')
        for bbox in entry['annotations']:
            cls_set.add(class_mapping[bbox['class']])
        return cls_set


class VatsalSrivastavaTrafficLightsChurchLotDataset(Dataset):

    def __init__(self):
        super(VatsalSrivastavaTrafficLightsChurchLotDataset, self).__init__(
            'vatsal_srivastava_traffic_lights_church_lot')
        self._labels = "real_data_annotations.yaml"

        self.label_set = {'Green', 'Yellow', 'Red'}

        self._singular_class_mapping = {label: self.get_present_label() for label in self.label_set}

        self._ternary_class_mapping = {label: self._choose_label(label) for label in self.label_set}

    @classmethod
    def _choose_label(cls, name):
        if name.startswith('Red'):
            return cls.get_red_label()
        elif name.startswith('Yellow'):
            return cls.get_yellow_label()
        elif name.startswith('Green'):
            return cls.get_green_label()
        else:
            raise ValueError("unknown label name: " + name)

    def get_all_labels(self, input_dir: str) -> list:
        with open(os.path.join(input_dir, self._labels)) as f:
            labels = yaml.safe_load(f)
        return labels

    def filter_original_labels(self, labels_file_content: list) -> list:
        """
        No need to filter anything here.
        :param labels_file_content: original labels content
        :return: filtered labels content
        """
        return labels_file_content

    def get_class_mapping(self, mode: str):
        if mode == 'singular':
            return self._singular_class_mapping
        elif mode == 'ternary':
            return self._ternary_class_mapping
        else:
            raise ValueError("unknown mode: " + mode)

    def get_output_labels_line(self, entry, bboxes: np.ndarray, output_image_path: str, mode: str) -> str:
        class_mapping = self.get_class_mapping(mode)
        line = output_image_path
        assert bboxes.shape[0] == len(entry['annotations'])
        for i in range(bboxes.shape[0]):
            line += ' ' + str(int(round(bboxes[i][0]))) + ',' \
                        + str(int(round(bboxes[i][1]))) + ',' \
                        + str(int(round(bboxes[i][2]))) + ',' \
                        + str(int(round(bboxes[i][3]))) + ',' \
                        + str(class_mapping[entry['annotations'][i]['class']])
        return line

    def get_input_image_path(self, input_dir, entity):
        return os.path.join(input_dir, entity['filename'])

    def get_bounding_boxes(self, entry, orig_image_shape):
        bboxes = []
        for bbox in entry['annotations']:
            bboxes.append([bbox['xmin'], bbox['ymin'], bbox['xmin'] + bbox['x_width'], bbox['ymin'] + bbox['y_height']])
        return np.asarray(bboxes)

    def get_light_counters(self, entry) -> Tuple[int, int, int, int]:
        class_mapping = self.get_class_mapping('ternary')
        counters = [0, 0, 0, 0]
        for bbox in entry['annotations']:
            counters[class_mapping[bbox['class']]] += 1
        if not entry['annotations']:
            counters[3] += 1
        return counters[0], counters[1], counters[2], counters[3]

    def get_entry_classes(self, entry):
        cls_set = set()
        class_mapping = self.get_class_mapping('ternary')
        for bbox in entry['annotations']:
            cls_set.add(class_mapping[bbox['class']])
        return cls_set


class YoloMarkDataset(Dataset):

    def __init__(self):
        super(YoloMarkDataset, self).__init__('yolo_mark')
        self.label_set = {0, 1, 2}

        self._singular_class_mapping = {label: self.get_present_label() for label in self.label_set}

        self._ternary_class_mapping = {label: label for label in self.label_set}

    def get_all_labels(self, input_dir: str) -> list:
        label_paths = glob(os.path.join(input_dir, "*.txt"))
        labels = []
        for path in label_paths:
            entry = [os.path.basename(path.replace(".txt", ".jpg"))]
            with open(path, 'r') as f:
                f_content = f.readlines()
                for line in f_content:
                    line_elems = line.strip().split(' ')
                    if line_elems:
                        cls, x_center, y_center, width, height = line_elems
                        entry.append([int(cls), float(x_center), float(y_center), float(width), float(height)])
            labels.append(entry)
        return labels

    def filter_original_labels(self, labels_file_content: list) -> list:
        filtered_labels = []
        for entry in labels_file_content:
            has_unknown = False
            for box in entry[1:]:
                if box[0] == 3:
                    has_unknown = True
            if not has_unknown:
                filtered_labels.append(entry)
        return filtered_labels

    def get_class_mapping(self, mode: str):
        if mode == 'singular':
            return self._singular_class_mapping
        elif mode == 'ternary':
            return self._ternary_class_mapping
        else:
            raise ValueError("unknown mode: " + mode)

    def get_output_labels_line(self, entry, bboxes: np.ndarray, output_image_path: str, mode: str) -> str:
        class_mapping = self.get_class_mapping(mode)
        line = output_image_path
        assert bboxes.shape[0] == (len(entry)-1)
        for i in range(bboxes.shape[0]):
            line += ' ' + str(int(round(bboxes[i][0]))) + ',' \
                        + str(int(round(bboxes[i][1]))) + ',' \
                        + str(int(round(bboxes[i][2]))) + ',' \
                        + str(int(round(bboxes[i][3]))) + ',' \
                        + str(class_mapping[entry[i+1][0]])
        return line

    def get_input_image_path(self, input_dir, entry):
        return os.path.join(input_dir, entry[0])

    def get_bounding_boxes(self, entry, orig_image_shape):
        h, w = orig_image_shape[0:2]
        bboxes = []
        for bbox in entry[1:]:
            x_center = w * bbox[1]
            y_center = h * bbox[2]
            x_side = w * bbox[3]
            y_side = h * bbox[4]
            bboxes.append([x_center - x_side/2, y_center - y_side/2, x_center + x_side/2, y_center + y_side/2])
        return np.asarray(bboxes)

    def get_light_counters(self, entry) -> Tuple[int, int, int, int]:
        class_mapping = self.get_class_mapping('ternary')
        counters = [0, 0, 0, 0]
        for bbox in entry[1:]:
            counters[class_mapping[bbox[0]]] += 1
        if not entry[1:]:
            counters[3] += 1
        return counters[0], counters[1], counters[2], counters[3]

    def get_entry_classes(self, entry):
        cls_set = set()
        class_mapping = self.get_class_mapping('ternary')
        for bbox in entry[1:]:
            cls_set.add(class_mapping[bbox[0]])
        return cls_set


KNOWN_DATASETS = {
    'bosch_small_traffic_lights': BoschSmallTrafficLightsDataset(),
    'vatsal_srivastava_traffic_lights_simulator': VatsalSrivastavaTrafficLightsSimulatorDataset(),
    'vatsal_srivastava_traffic_lights_church_lot': VatsalSrivastavaTrafficLightsChurchLotDataset(),
    'yolo_mark': YoloMarkDataset(),
}


class DataPreparer:

    def __init__(self, dataset: Dataset, fliplr: bool, scale: bool, balance:bool,
                 input_dir: str, output_dir: str, continue_output_dir: bool, draw_bounding_boxes: bool):
        self.dataset = dataset
        self.balance = balance
        self.draw_bounding_boxes = draw_bounding_boxes
        self.transforms = [self._noop]
        if fliplr:
            self.transforms.append(self._fliplr)
        if scale:
            self.transforms.append(self._scale)
            if fliplr:
                self.transforms.append(self._fliplr_and_scale)

        if not os.path.isdir(input_dir):
            raise FileNotFoundError(input_dir + " directory does not exist")
        self.input_dir = input_dir

        if not continue_output_dir and os.path.exists(output_dir):
            raise FileExistsError(output_dir + " file or directory exists")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=continue_output_dir)
        os.makedirs(self.dataset.get_output_images_dir(output_dir), exist_ok=continue_output_dir)
        os.makedirs(self.dataset.get_output_labels_dir(output_dir), exist_ok=continue_output_dir)

        print("Dataset:         ", self.dataset.name)
        print("Input Directory: ", self.input_dir)
        print("Output Directory:", self.output_dir)

        self.original_labels = None

    def _read_original_labels(self):
        self.original_labels = self.dataset.get_all_labels(self.input_dir)

    def get_entries_containing_label(self, label):
        entries = []
        filtered_labels = self.dataset.filter_original_labels(self.original_labels)
        for entry in filtered_labels:
            clses = self.dataset.get_entry_classes(entry)
            if (label in clses) or (label is None and len(clses) == 0):
                entries.append(entry)
        return entries

    @staticmethod
    def _to_vatsal_yaml_line(line):
        class_map =  {'0': 'Red', '1': 'Yellow', '2': 'Green'}
        line_parts = line.strip().split(' ')
        annotations = []
        for box in line_parts[1:]:
            xmin, ymin, xmax, ymax, cls = box.split(',')
            annotations.append({'class': class_map[cls], 'x_width': int(xmax)-int(xmin), 'xmin': int(xmin),
                                'y_height': int(ymax)-int(ymin), 'ymin': int(ymin)})

        return [{'filename': line_parts[0], 'class': 'image', 'annotations': annotations}]

    @staticmethod
    def _to_bosh_yaml_line(line):
        class_map =  {'0': 'Red', '1': 'Yellow', '2': 'Green'}
        line_parts = line.strip().split(' ')
        annotations = []
        for box in line_parts[1:]:
            xmin, ymin, xmax, ymax, cls = box.split(',')
            annotations.append({'label': class_map[cls], 'occluded': False,
                                'x_max': int(xmax), 'xmin': int(xmin), 'y_max': int(ymax), 'ymin': int(ymin)})

        return [{'path': line_parts[0], 'boxes': annotations}]

    @staticmethod
    def _to_yolo_mark_file_content(line, img_shape):
        h, w = img_shape[0:2]
        line_parts = line.strip().split(' ')
        out_lines = []
        for box in line_parts[1:]:
            xmin, ymin, xmax, ymax, cls = box.split(',')
            x_center = (float(xmax) + float(xmin)) / 2 / w
            x_side = (float(xmax) - float(xmin)) / w
            y_center = (float(ymax) + float(ymin)) / 2 / h
            y_side = (float(ymax) - float(ymin)) / h
            out_lines.append("%s %s %s %s %s" % (cls, x_center, y_center, x_side, y_side))

        return '\n'.join(out_lines)

    @staticmethod
    def _ndarray_to_BoundingBoxesOnImage(bboxes: np.ndarray, img_shape) -> ia.BoundingBoxesOnImage:
        bb_list = []
        for bbox in bboxes:
            bb_list.append(ia.BoundingBox(*bbox))

        return ia.BoundingBoxesOnImage(bb_list, shape=img_shape)

    @staticmethod
    def _BoundingBoxesOnImage_to_ndarray(bboxes_on_image: ia.BoundingBoxesOnImage):
        bb_list = []
        for bbox in bboxes_on_image.bounding_boxes:
            bb_list.append([bbox.x1, bbox.y1, bbox.x2, bbox.y2])
        return np.asarray(bb_list)

    @classmethod
    def _fliplr(cls, image: np.ndarray, bboxes: np.ndarray) -> Tuple[np.ndarray, ia.BoundingBoxesOnImage]:
        seq = iaa.Sequential([
            iaa.Fliplr(1.0),  # horizontally flip
        ])

        bbs = cls._ndarray_to_BoundingBoxesOnImage(bboxes, image.shape)

        # Make our sequence deterministic.
        # We can now apply it to the image and then to the BBs and it will lead to the same augmentations.
        seq_det = seq.to_deterministic()

        image_aug = seq_det.augment_images([image])[0]
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

        return image_aug, bbs_aug

    @classmethod
    def _scale(cls, image: np.ndarray, bboxes: np.ndarray) -> Tuple[np.ndarray, ia.BoundingBoxesOnImage]:
        seq = iaa.Sequential([
            iaa.Affine(scale=0.7, mode='edge'),  # scale image, preserving original image shape
        ])

        bbs = cls._ndarray_to_BoundingBoxesOnImage(bboxes, image.shape)

        # Make our sequence deterministic.
        # We can now apply it to the image and then to the BBs and it will lead to the same augmentations.
        seq_det = seq.to_deterministic()

        image_aug = seq_det.augment_images([image])[0]
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

        return image_aug, bbs_aug

    @classmethod
    def _fliplr_and_scale(cls, image: np.ndarray, bboxes: np.ndarray) -> Tuple[np.ndarray, ia.BoundingBoxesOnImage]:
        seq = iaa.Sequential([
            iaa.Fliplr(1.0),  # horizontally flip
            iaa.Affine(scale=0.7, mode='edge'),  # scale image, preserving original image shape
        ])

        bbs = cls._ndarray_to_BoundingBoxesOnImage(bboxes, image.shape)

        # Make our sequence deterministic.
        # We can now apply it to the image and then to the BBs and it will lead to the same augmentations.
        seq_det = seq.to_deterministic()

        image_aug = seq_det.augment_images([image])[0]
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

        return image_aug, bbs_aug

    @classmethod
    def _noop(cls, image: np.ndarray, bboxes: np.ndarray) -> Tuple[np.ndarray, ia.BoundingBoxesOnImage]:
        return image.copy(), cls._ndarray_to_BoundingBoxesOnImage(bboxes, image.shape)

    @classmethod
    def _random_transforms(cls, image: np.ndarray, bboxes: np.ndarray):
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),  # horizontal flips
            # Strengthen or weaken the contrast in each image.
            iaa.ContrastNormalization((0.75, 1.5)),
            # Make some images brighter and some darker.
            iaa.Multiply((0.8, 1.2)),
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(scale=(0.8, 1.0), mode='edge')
        ], random_order=True)  # apply augmenters in random order

        bbs = cls._ndarray_to_BoundingBoxesOnImage(bboxes, image.shape)

        seq_det = seq.to_deterministic()

        image_aug = seq_det.augment_images([image])[0]
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

        return image_aug, bbs_aug

    def balance_dataset(self, red_counter, yellow_counter, green_counter, nolight_counter):
        if 0 in [red_counter, yellow_counter, green_counter, nolight_counter]:
            raise ValueError('cannot balance dataset where some traffic light classes have no representatives')
        print('\n\nStart balancing dataset...\n')

        max_between_classes = max(red_counter, yellow_counter, green_counter)
        if (nolight_counter > 3 * max_between_classes):
            val = nolight_counter // 3
            target_red_cnt, target_yellow_cnt, target_green_cnt, target_nolight_cnt = val, val, val, nolight_counter
        else:
            target_red_cnt, target_yellow_cnt, target_green_cnt, target_nolight_cnt = \
                max_between_classes, max_between_classes, max_between_classes, 3*max_between_classes

        red_entries = [random.choice(self.get_entries_containing_label(self.dataset.get_red_label()))
                                     for _ in range(target_red_cnt - red_counter)]
        yellow_entries = [random.choice(self.get_entries_containing_label(self.dataset.get_yellow_label()))
                                     for _ in range(target_yellow_cnt - yellow_counter)]
        green_entries = [random.choice(self.get_entries_containing_label(self.dataset.get_green_label()))
                                     for _ in range(target_green_cnt - green_counter)]
        nolight_entries = [random.choice(self.get_entries_containing_label(None))
                                     for _ in range(target_nolight_cnt - nolight_counter)]

        transforms = [self._random_transforms]
        entries = []
        entries.extend(red_entries)
        entries.extend(yellow_entries)
        entries.extend(green_entries)
        entries.extend(nolight_entries)

        red_cnt, yellow_cnt, green_cnt, nolight_cnt = self.process_data(transforms, entries)

        assert target_red_cnt     == red_cnt     + red_counter
        assert target_yellow_cnt  == yellow_cnt  + yellow_counter
        assert target_green_cnt   == green_cnt   + green_counter
        assert target_nolight_cnt == nolight_cnt + nolight_counter

        return target_red_cnt, target_yellow_cnt, target_green_cnt, target_nolight_cnt

    def process_data(self, transforms, filtered_entries):
        counter = 0
        red_counter = 0
        yellow_counter = 0
        green_counter = 0
        nolight_counter = 0
        for aug_func in transforms:

            with open(self.dataset.get_output_labels_singular_file_name(self.output_dir), 'a+') as f_singular, \
                    open(self.dataset.get_output_labels_ternary_file_name(self.output_dir), 'a+') as f_ternary, \
                    open(self.dataset.get_output_labels_ternary_vatsal_yaml_file_name(self.output_dir), 'a+') \
                            as f_vatsal_yaml_ternary, \
                    open(self.dataset.get_output_labels_ternary_bosh_yaml_file_name(self.output_dir), 'a+') \
                            as f_bosh_yaml_ternary:
                for entry in filtered_entries:
                    counter += 1
                    in_img_path = self.dataset.get_input_image_path(self.input_dir, entry)
                    out_img_path = self.dataset.get_output_image_path(self.output_dir, in_img_path)

                    # read input image and extract corresponding bounding boxes
                    in_img = cv2.imread(in_img_path)
                    in_bboxes = self.dataset.get_bounding_boxes(entry, in_img.shape)

                    out_img, out_bboxes_on_images = aug_func(in_img, in_bboxes)
                    out_bboxes = self._BoundingBoxesOnImage_to_ndarray(out_bboxes_on_images)

                    # optionally, draw rectangles (useful for verifying the script correctness)
                    if self.draw_bounding_boxes:
                        out_img = out_bboxes_on_images.draw_on_image(out_img, thickness=2, color=(255, 255, 255))

                    # get label line for singular and ternary output label files
                    line_singular = self.dataset.get_output_labels_line(entry, out_bboxes, out_img_path, 'singular') \
                                    + '\n'
                    line_ternary = \
                        self.dataset.get_output_labels_line(entry, out_bboxes, out_img_path, 'ternary') + '\n'

                    # write output image and label files
                    cv2.imwrite(out_img_path, out_img)
                    f_singular.write(line_singular)
                    f_ternary.write(line_ternary)

                    # write yaml files
                    yaml.dump(self._to_vatsal_yaml_line(line_ternary), f_vatsal_yaml_ternary)
                    yaml.dump(self._to_bosh_yaml_line(line_ternary), f_bosh_yaml_ternary)

                    # write Yolo_mark format
                    with open(self.dataset.get_output_labels_ternary_yolo_mark_image_file_name(out_img_path), 'w+') \
                            as f_yolo_mark_ternary:
                        f_yolo_mark_ternary.write(self._to_yolo_mark_file_content(line_ternary, out_img.shape))

                    red_cnt, yellow_cnt, green_cnt, nolight_cnt = self.dataset.get_light_counters(entry)
                    red_counter += red_cnt
                    yellow_counter += yellow_cnt
                    green_counter += green_cnt
                    nolight_counter += nolight_cnt
                    print("\n"
                          "image number: {img_num}\n"
                          "input image: {in_img_path}\n"
                          "(red,yellow,green,nolight): ({red},{yellow},{green},{nolight})\n"
                          "input bounding boxes:\n{in_bb}\n"
                          "output image: {out_img_path}\n"
                          "output bounding boxes:\n{out_bb}\n".format(img_num=counter, in_img_path=in_img_path,
                                                                      in_bb=in_bboxes, out_img_path=out_img_path,
                                                                      out_bb=out_bboxes,
                                                                      red=red_cnt, yellow=yellow_cnt, green=green_cnt,
                                                                      nolight=nolight_cnt))

        return red_counter, yellow_counter, green_counter, nolight_counter

    def write_statistics(self, red_counter, yellow_counter, green_counter, nolight_counter):
        # update counters in accordance with what already in the statistics file
        if os.path.exists(self.dataset.get_statistics_file_name(self.output_dir)):
            with open(self.dataset.get_statistics_file_name(self.output_dir), 'r') as f_stat:
                lines = f_stat.readlines()
            for line in lines:
                val = int(line.strip().split(':')[-1].strip())
                if line.startswith('red'):
                    red_counter += val
                elif line.startswith('yellow'):
                    yellow_counter += val
                elif line.startswith('green'):
                    green_counter += val
                elif line.startswith('nolight'):
                    nolight_counter += val
                else:
                    raise IOError('file ' + self.dataset.get_statistics_file_name(self.output_dir)
                                  + ' has a content of unknown format')
        # write counters to statistics file
        stat_info = "red:     %s\n" \
                    "yellow:  %s\n" \
                    "green:   %s\n" \
                    "nolight: %s\n" % (red_counter, yellow_counter, green_counter, nolight_counter)
        print('DATASET STATISTICS:')
        print(stat_info)
        with open(self.dataset.get_statistics_file_name(self.output_dir), 'w+') as f_stat:
            f_stat.write(stat_info)

    def prepare(self):
        if self.original_labels is None:
            self._read_original_labels()
        filtered_labels = self.dataset.filter_original_labels(self.original_labels)

        print("Entries in original dataset:", len(self.original_labels))
        print("Entries in filtered dataset", len(filtered_labels))

        red_counter, yellow_counter, green_counter, nolight_counter = \
            self.process_data(self.transforms, filtered_labels)

        # write counters to statistics file
        stat_info = "red:     %s\n" \
                    "yellow:  %s\n" \
                    "green:   %s\n" \
                    "nolight: %s\n" % (red_counter, yellow_counter, green_counter, nolight_counter)
        print('DURING THIS RUN IDENTIFIED:')
        print(stat_info)

        if self.balance:
            red_counter, yellow_counter, green_counter, nolight_counter = \
                self.balance_dataset(red_counter, yellow_counter, green_counter, nolight_counter)

        self.write_statistics(red_counter, yellow_counter, green_counter, nolight_counter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=\
"""\
This script is capable of working with several datasets from the list below. 
It applies the requested image augmentation to the images from the provided dataset
and converts labels to several formats specified below. It also balances dataset to the following
form: red == yellow == green == nolight/3.

Datasets:
    - Bosh Small Traffic Lights Dataset: https://hci.iwr.uni-heidelberg.de/node/6132
    - Vatsal Srivastava's Traffic Lights Dataset (Simulator & Church Lot):  
          https://drive.google.com/file/d/0B-Eiyn-CUQtxdUZWMkFfQzdObUE/view?usp=sharing
    - Any Traffic Lights Dataset Labeled with Yolo_mark: https://github.com/AlexeyAB/Yolo_mark

Label formats:
    - One row for one image (singular and ternary); 
      Row format: image_file_path box1 box2 ... boxN; 
      Box format: x_min,y_min,x_max,y_max,class_id (no space).
    - Vatsal Srivastava's yaml format (only ternary). Example:
      - annotations:
        - {class: Green, x_width: 17, xmin: 298, y_height: 49, ymin: 153}
        class: image
        filename: ./images/a0a05c4e-b2be-4a85-aebd-93f0e78ff3b7.jpg
      - annotations:
        - {class: Yellow, x_width: 15, xmin: 364, y_height: 43, ymin: 156}
        - {class: Yellow, x_width: 15, xmin: 151, y_height: 52, ymin: 100}
        class: image
        filename: ./images/ccbd292c-89cb-4e8b-a671-47b57ebb672b.jpg
    - Bosh Small Traffic Lights yaml format (only ternary). Example:
      - boxes:
        - {label: Red, occluded: false, x_max: 640, x_min: 633, y_max: 355, y_min: 344}
        - {label: Yellow, occluded: false, x_max: 659, x_min: 651, y_max: 366, y_min: 353}
        path: ./images/ccbd292c-89cb-4e8b-a671-47b57ebb672b.png
    - Yolo_mark format. One file per image. Example: image_name.jpg -> image_name.txt. Content:
      <object-class> <x_center> <y_center> <width> <height>
      <object-class> <x_center> <y_center> <width> <height>
      ...
""",
        add_help=True,
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('--dataset', action='store',  type=str, required=True,
                        choices=['bosch_small_traffic_lights', 'vatsal_srivastava_traffic_lights_simulator',
                                 'vatsal_srivastava_traffic_lights_church_lot', 'yolo_mark'],
                        help='dataset name')
    parser.add_argument('--fliplr', action='store_true',
                        help="apply imgaug.Fliplr function (flip horizontally) to all images; "
                             "dataset size will x2 in size")
    parser.add_argument('--scale', action='store_true',
                        help="apply imgaug.Affine(scale=(0.7, 0.7)) function "
                             "(scale image, keeping original image shape); dataset size will x2 in size")
    parser.add_argument('--balance', action='store_true',
                        help="balance dataset, so that there is an equal number of representatives of each class")
    parser.add_argument('--input-dir', action='store', type=str, required=True, metavar='DIR',
                        help="dataset's root directory")
    parser.add_argument('--output-dir', action='store', type=str, required=True, metavar='DIR',
                        help="directory to store prepared images and labels")
    parser.add_argument('--continue-output-dir', action='store_true',
                        help="expand existing output directory with new image-label entries")
    parser.add_argument('--draw-bounding-boxes', action='store_true',
                        help="draw bounding boxes on the output images; "
                             "do not use it while preparing data for training")

    args = parser.parse_args()

    DataPreparer(dataset=KNOWN_DATASETS[args.dataset],
                 fliplr=args.fliplr,
                 scale=args.scale,
                 balance=args.balance,
                 input_dir=args.input_dir,
                 output_dir=args.output_dir,
                 continue_output_dir=args.continue_output_dir,
                 draw_bounding_boxes=args.draw_bounding_boxes).prepare()
