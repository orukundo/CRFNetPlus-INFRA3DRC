# -*- coding: utf-8 -*-
"""
INFRA-3DRC data generator for the public CRF-Net implementation.

This file is intentionally Python-3.5-compatible because the original CRF-Net
repository was published for Python 3.5 / TensorFlow 1.13 / CUDA 10.

It does NOT require the modern infra-3drc Python package. Instead, it reads the
public dataset layout directly, using the calibration matrices supplied in each
scene's calibration.json.

Input to CRF-Net:
    channels 0,1,2 : camera B,G,R image
    channel 3      : radar range
    channel 4      : radar range_rate (Doppler)
    channel 5      : radar RCS

Radar points are projected to the mono-camera image using:
    uvw = K [R|t] [x y z 1]^T

Camera annotations are used as 2D detection ground truth.
"""

from __future__ import division

import ast
import json
import os
import random
import struct

import cv2
import numpy as np

from .generator import Generator


class Infra3DRCGenerator(Generator):
    DATATYPE = np.float32

    CATEGORY_ID_TO_NAME = {
        1: 'adult',
        2: 'child',
        3: 'group',
        4: 'bicycle',
        5: 'motorcycle',
        6: 'car',
        7: 'bus',
        8: 'truck',
    }

    def __init__(
        self,
        dataset_root,
        scene_numbers,
        split='train',
        split_ratios=(0.70, 0.15, 0.15),
        seed=42,
        channels=None,
        category_mapping=None,
        camera_dropout=0.0,
        radar_dropout=0.0,
        normalize_radar=False,
        sample_selection=False,
        only_radar_annotated=False,
        n_sweeps=1,
        noise_filter=None,
        noise_filter_threshold=0.0,
        noisy_image_method=None,
        noise_factor=0.0,
        perfect_noise_filter=False,
        noise_category_selection=None,
        inference=False,
        **kwargs
    ):
        if channels is None:
            channels = [0, 1, 2, 3, 4, 5]

        self.dataset_root = os.path.abspath(dataset_root)
        self.scene_numbers = [int(x) for x in scene_numbers]
        self.split = split
        self.split_ratios = split_ratios
        self.seed = int(seed)
        self.channels = list(channels)
        self.camera_dropout = float(camera_dropout)
        self.radar_dropout = float(radar_dropout)
        self.normalize_radar = normalize_radar
        self.sample_selection = sample_selection
        self.only_radar_annotated = int(only_radar_annotated) if only_radar_annotated else 0
        self.n_sweeps = n_sweeps
        self.inference = inference

        # Keep the same metadata expected by CRF-Net's generator interface.
        self.labels = {}
        self.classes, self.labels = self._get_class_label_mapping(
            list(self.CATEGORY_ID_TO_NAME.values()),
            category_mapping,
        )

        self.image_min_side = kwargs['image_min_side']
        self.image_max_side = kwargs['image_max_side']

        self.samples = []
        self.scene_calibrations = {}

        for scene_number in self.scene_numbers:
            scene_path = self._resolve_scene_path(scene_number)
            calib = self._load_calibration(scene_path)
            self.scene_calibrations[scene_number] = calib

            images = self._sorted_files(
                os.path.join(scene_path, 'camera_01', 'camera_01__data'),
                suffix='.png'
            )
            camera_ann = self._sorted_files(
                os.path.join(scene_path, 'camera_01', 'camera_01__annotation'),
                suffix='.json'
            )
            radar_pcd = self._sorted_files(
                os.path.join(scene_path, 'radar_01', 'radar_01__data'),
                suffix='.pcd'
            )
            radar_ann = self._sorted_files(
                os.path.join(scene_path, 'radar_01', 'radar_01__annotation'),
                suffix='.json'
            )

            lengths = [len(images), len(camera_ann), len(radar_pcd), len(radar_ann)]
            if len(set(lengths)) != 1:
                raise RuntimeError(
                    'Scene {:02d} synchronized file counts differ: {}'.format(
                        scene_number, lengths
                    )
                )

            frame_indices = list(range(len(images)))
            rng = random.Random(self.seed + scene_number)
            rng.shuffle(frame_indices)

            n = len(frame_indices)
            n_train = int(round(self.split_ratios[0] * n))
            n_val = int(round(self.split_ratios[1] * n))
            n_train = min(n_train, max(0, n - 2))
            n_val = min(n_val, max(0, n - n_train - 1))

            if split == 'train':
                selected = frame_indices[:n_train]
            elif split in ('val', 'validation'):
                selected = frame_indices[n_train:n_train + n_val]
            elif split == 'test':
                selected = frame_indices[n_train + n_val:]
            elif split in ('all', None):
                selected = frame_indices
            else:
                raise ValueError('Unknown split: {}'.format(split))

            selected.sort()

            for idx in selected:
                self.samples.append({
                    'scene_number': scene_number,
                    'frame_index': idx,
                    'image_path': images[idx],
                    'camera_annot_path': camera_ann[idx],
                    'radar_pcd_path': radar_pcd[idx],
                    'radar_annot_path': radar_ann[idx],
                })

        if len(self.samples) == 0:
            raise RuntimeError('No INFRA-3DRC samples found for split {}'.format(split))

        super(Infra3DRCGenerator, self).__init__(**kwargs)

    @staticmethod
    def _get_class_label_mapping(category_names, category_mapping):
        """Exact source-name -> target-name mapping for INFRA-3DRC."""
        if category_mapping is None:
            category_mapping = dict((name, name) for name in category_names)

        # ConfigParser produces strings for keys/values.
        clean_mapping = {}
        for source_name, target_name in category_mapping.items():
            source_name = str(source_name).strip()
            target_name = str(target_name).strip()
            if source_name in category_names and target_name:
                clean_mapping[source_name] = target_name

        target_names = sorted(set(clean_mapping.values()))
        label_to_name = dict((i, name) for i, name in enumerate(target_names))
        label_to_name[len(label_to_name)] = 'bg'

        name_to_label = {}
        for source_name, target_name in clean_mapping.items():
            name_to_label[source_name] = target_names.index(target_name)

        return name_to_label, label_to_name

    def _resolve_scene_path(self, scene_number):
        scene_name = 'INFRA-3DRC_scene-{:02d}'.format(scene_number)

        candidates = [
            os.path.join(self.dataset_root, scene_name),
            os.path.join(self.dataset_root, scene_name, scene_name),
        ]

        # Also allow data_path itself to point directly to one scene folder.
        if os.path.basename(self.dataset_root.rstrip(os.sep)) == scene_name:
            candidates.insert(0, self.dataset_root)

        # Do not accept a directory merely because it exists. The downloaded
        # INFRA-3DRC archives often contain a duplicated scene folder, e.g.
        # INFRA-3DRC_scene-01/INFRA-3DRC_scene-01/.  We therefore choose the
        # first candidate that actually contains the sensor folders/files.
        for candidate in candidates:
            camera_dir = os.path.join(candidate, 'camera_01')
            radar_dir = os.path.join(candidate, 'radar_01')
            calibration = os.path.join(candidate, 'calibration.json')

            if (
                os.path.isdir(camera_dir)
                and os.path.isdir(radar_dir)
                and os.path.isfile(calibration)
            ):
                return candidate

        raise IOError(
            'Could not find a valid {} folder below {}. Expected camera_01, '
            'radar_01 and calibration.json.'.format(scene_name, self.dataset_root)
        )

    @staticmethod
    def _sorted_files(folder, suffix=None):
        if not os.path.isdir(folder):
            raise IOError('Directory not found: {}'.format(folder))

        files = []
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                continue
            if suffix is not None and not name.lower().endswith(suffix):
                continue
            files.append(path)
        files.sort()
        return files

    @staticmethod
    def _load_json(path):
        with open(path, 'r') as handle:
            return json.load(handle)

    @staticmethod
    def _load_calibration(scene_path):
        path = os.path.join(scene_path, 'calibration.json')
        data = Infra3DRCGenerator._load_json(path)

        camera_intrinsics = None
        radar_to_camera = None

        for entry in data['calibration']:
            name = entry['calibration']
            if name == 'camera_01':
                camera_intrinsics = np.asarray(entry['k'], dtype=np.float32)
            elif name == 'radar_01_to_camera_01':
                radar_to_camera = np.asarray(entry['T'], dtype=np.float32)

        if camera_intrinsics is None or radar_to_camera is None:
            raise RuntimeError(
                'Missing camera_01 or radar_01_to_camera_01 calibration in {}'.format(path)
            )

        camera_intrinsics = camera_intrinsics.reshape((3, 3))
        radar_to_camera = radar_to_camera.reshape((3, 4))

        return {
            'K': camera_intrinsics,
            'T': radar_to_camera,
            'P': np.dot(camera_intrinsics, radar_to_camera),
        }

    @staticmethod
    def _pcd_numpy_dtype(type_char, size):
        key = (type_char.upper(), int(size))
        mapping = {
            ('F', 4): '<f4',
            ('F', 8): '<f8',
            ('I', 1): '<i1',
            ('I', 2): '<i2',
            ('I', 4): '<i4',
            ('I', 8): '<i8',
            ('U', 1): '<u1',
            ('U', 2): '<u2',
            ('U', 4): '<u4',
            ('U', 8): '<u8',
        }
        if key not in mapping:
            raise ValueError('Unsupported PCD TYPE/SIZE: {}'.format(key))
        return mapping[key]

    @staticmethod
    def _read_pcd(path):
        """Read binary or ASCII PCD into a NumPy structured array."""
        metadata = {}
        data_offset = None

        with open(path, 'rb') as handle:
            while True:
                line = handle.readline()
                if not line:
                    break
                decoded = line.decode('utf-8').strip()
                if not decoded or decoded.startswith('#'):
                    continue
                parts = decoded.split()
                key = parts[0].upper()
                values = parts[1:]
                metadata[key] = values
                if key == 'DATA':
                    data_offset = handle.tell()
                    break

            if data_offset is None:
                raise ValueError('PCD DATA line not found: {}'.format(path))

            fields = metadata['FIELDS']
            sizes = [int(v) for v in metadata['SIZE']]
            types = metadata['TYPE']
            counts = [int(v) for v in metadata.get('COUNT', ['1'] * len(fields))]
            point_count = int(metadata.get('POINTS', metadata.get('WIDTH', ['0']))[0])
            data_mode = metadata['DATA'][0].lower()

            dtype_fields = []
            for field, size, typ, count in zip(fields, sizes, types, counts):
                np_type = Infra3DRCGenerator._pcd_numpy_dtype(typ, size)
                if count == 1:
                    dtype_fields.append((field, np_type))
                else:
                    dtype_fields.append((field, np_type, (count,)))

            dtype = np.dtype(dtype_fields)

            if data_mode == 'binary':
                raw = handle.read()
                cloud = np.frombuffer(raw, dtype=dtype, count=point_count).copy()
            elif data_mode == 'ascii':
                handle.seek(data_offset)
                text = handle.read().decode('utf-8')
                rows = [row.split() for row in text.splitlines() if row.strip()]
                cloud = np.zeros((len(rows),), dtype=dtype)
                for row_index, row in enumerate(rows):
                    cursor = 0
                    for field, count in zip(fields, counts):
                        if count == 1:
                            cloud[field][row_index] = float(row[cursor])
                        else:
                            cloud[field][row_index] = [float(v) for v in row[cursor:cursor + count]]
                        cursor += count
            else:
                raise ValueError(
                    'PCD mode {} is not supported (binary_compressed not implemented).'.format(data_mode)
                )

        # Match the public SDK: radar annotations are available to 120 m in x.
        if 'x' in cloud.dtype.names:
            cloud = cloud[cloud['x'] <= 120.0]

        return cloud

    @staticmethod
    def _field_or(cloud, name, default=0.0):
        if name in cloud.dtype.names:
            return np.asarray(cloud[name], dtype=np.float32)
        return np.full((len(cloud),), float(default), dtype=np.float32)

    @staticmethod
    def _project_radar(cloud, calibration, image_shape):
        if len(cloud) == 0:
            return np.zeros((0,), dtype=[
                ('u', '<f4'), ('v', '<f4'), ('range', '<f4'),
                ('range_rate', '<f4'), ('rcs', '<f4')
            ])

        x = Infra3DRCGenerator._field_or(cloud, 'x')
        y = Infra3DRCGenerator._field_or(cloud, 'y')
        z = Infra3DRCGenerator._field_or(cloud, 'z')

        xyz1 = np.stack(
            [x, y, z, np.ones_like(x, dtype=np.float32)],
            axis=1
        )
        uvw = np.dot(xyz1, calibration['P'].T)
        w = uvw[:, 2]

        valid = np.isfinite(w) & (w > 1e-6)
        u = np.zeros_like(w, dtype=np.float32)
        v = np.zeros_like(w, dtype=np.float32)
        u[valid] = uvw[valid, 0] / w[valid]
        v[valid] = uvw[valid, 1] / w[valid]

        h, width = image_shape[:2]
        valid = valid & np.isfinite(u) & np.isfinite(v)
        valid = valid & (u >= 0) & (u < width) & (v >= 0) & (v < h)

        range_value = Infra3DRCGenerator._field_or(cloud, 'range')
        if 'range' not in cloud.dtype.names:
            range_value = np.sqrt(x * x + y * y + z * z)

        range_rate = Infra3DRCGenerator._field_or(cloud, 'range_rate')
        rcs = Infra3DRCGenerator._field_or(cloud, 'rcs')

        dtype = np.dtype([
            ('u', '<f4'), ('v', '<f4'), ('range', '<f4'),
            ('range_rate', '<f4'), ('rcs', '<f4')
        ])
        out = np.zeros((int(np.count_nonzero(valid)),), dtype=dtype)
        out['u'] = u[valid]
        out['v'] = v[valid]
        out['range'] = range_value[valid]
        out['range_rate'] = range_rate[valid]
        out['rcs'] = rcs[valid]
        return out

    @staticmethod
    def _linear_scale(value, low, high, out_low=-127.5, out_high=127.5):
        value = np.clip(value, low, high)
        if high <= low:
            return np.zeros_like(value)
        normalized = (value - low) / float(high - low)
        return normalized * (out_high - out_low) + out_low

    def _make_image_plus(self, sample):
        image = cv2.imread(sample['image_path'], cv2.IMREAD_COLOR)
        if image is None:
            raise IOError('Could not read image: {}'.format(sample['image_path']))

        # CRF-Net's original nuScenes generator creates image_plus at the final
        # configured HxW resolution before the generic Generator preprocessing.
        target_h = int(self.image_min_side)
        target_w = int(self.image_max_side)
        image_resized = cv2.resize(image, (target_w, target_h))

        if self.camera_dropout > 0.0 and np.random.rand() < self.camera_dropout:
            image_resized[:] = 0

        radar_maps = np.zeros((target_h, target_w, 3), dtype=np.float32)

        if not (self.radar_dropout > 0.0 and np.random.rand() < self.radar_dropout):
            cloud = self._read_pcd(sample['radar_pcd_path'])
            calibration = self.scene_calibrations[sample['scene_number']]
            projected = self._project_radar(cloud, calibration, image.shape)

            sx = target_w / float(image.shape[1])
            sy = target_h / float(image.shape[0])

            ranges = projected['range'].astype(np.float32)
            rates = projected['range_rate'].astype(np.float32)
            rcs = projected['rcs'].astype(np.float32)

            if self.normalize_radar:
                ranges = self._linear_scale(ranges, 0.0, 120.0)
                rates = self._linear_scale(rates, -30.0, 30.0)
                rcs = self._linear_scale(rcs, -30.0, 30.0)

            projection_height = max(1, int(round(self.radar_projection_height)))
            half = projection_height // 2

            # Draw farther points first so nearer measurements overwrite them.
            order = np.argsort(projected['range'])[::-1]

            for point_index in order:
                u = int(round(float(projected['u'][point_index]) * sx))
                v = int(round(float(projected['v'][point_index]) * sy))
                if u < 0 or u >= target_w or v < 0 or v >= target_h:
                    continue

                y0 = max(0, v - half)
                y1 = min(target_h, v + half + 1)
                radar_maps[y0:y1, u, 0] = ranges[point_index]
                radar_maps[y0:y1, u, 1] = rates[point_index]
                radar_maps[y0:y1, u, 2] = rcs[point_index]

        image_plus = np.concatenate(
            [image_resized.astype(np.float32), radar_maps],
            axis=2
        )
        return image_plus[:, :, self.channels]

    def size(self):
        return len(self.samples)

    def num_classes(self):
        return len(self.labels)

    def has_label(self, label):
        return label in self.labels

    def has_name(self, name):
        return name in self.classes

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def inv_label_to_name(self, name):
        inverse = dict((v, k) for k, v in self.labels.items())
        return inverse[name]

    def image_aspect_ratio(self, image_index):
        # image_plus is always resized to configured width/height.
        return self.image_max_side / float(self.image_min_side)

    def load_image(self, image_index):
        return self._make_image_plus(self.samples[image_index]).astype(self.DATATYPE)

    def load_annotations(self, image_index):
        sample = self.samples[image_index]
        camera_json = self._load_json(sample['camera_annot_path'])
        radar_json = self._load_json(sample['radar_annot_path'])

        # Obtain original camera resolution so boxes can be scaled exactly to
        # the HxW image_plus returned by load_image().
        image = cv2.imread(sample['image_path'], cv2.IMREAD_COLOR)
        if image is None:
            raise IOError('Could not read image: {}'.format(sample['image_path']))

        src_h, src_w = image.shape[:2]
        sx = self.image_max_side / float(src_w)
        sy = self.image_min_side / float(src_h)

        radar_by_det = {}
        for radar_obj in radar_json.get('objects', []):
            radar_by_det[radar_obj.get('det_id')] = radar_obj

        labels = []
        bboxes = []
        distances = []
        num_radar_pts = []
        visibilities = []

        for obj in camera_json.get('annotations', []):
            try:
                category_id = int(obj['category_id'])
            except Exception:
                continue

            source_name = self.CATEGORY_ID_TO_NAME.get(category_id)
            if source_name not in self.classes:
                continue

            bbox = obj.get('bbox')
            if bbox is None or len(bbox) != 4:
                continue

            x, y, w, h = [float(v) for v in bbox]
            if w <= 0 or h <= 0:
                continue

            x1 = max(0.0, min(float(src_w), x)) * sx
            y1 = max(0.0, min(float(src_h), y)) * sy
            x2 = max(0.0, min(float(src_w), x + w)) * sx
            y2 = max(0.0, min(float(src_h), y + h)) * sy

            if x2 <= x1 or y2 <= y1:
                continue

            radar_obj = radar_by_det.get(obj.get('det_id'))
            if radar_obj is not None:
                points = radar_obj.get('points', [])
            else:
                points = []

            if self.only_radar_annotated and len(points) == 0:
                continue

            labels.append(self.classes[source_name])
            bboxes.append([x1, y1, x2, y2])
            num_radar_pts.append(len(points))

            if points:
                # Public radar point rows are documented as including range.
                # Read metadata so we do not assume a column position.
                metadata = radar_json.get('radar_pcd_metadata', {})
                fields_text = metadata.get('fields', '[]')
                try:
                    fields = ast.literal_eval(fields_text)
                except Exception:
                    fields = []

                if 'range' in fields:
                    range_index = fields.index('range')
                    values = [float(p[range_index]) for p in points]
                    distances.append(float(np.mean(values)))
                else:
                    distances.append(0.0)
            else:
                distances.append(0.0)

            visibilities.append(0)

        annotations = {
            'labels': np.asarray(labels, dtype=np.int32),
            'bboxes': np.asarray(bboxes, dtype=np.float32).reshape((-1, 4)),
            'distances': np.asarray(distances, dtype=np.float32),
            'num_radar_pts': np.asarray(num_radar_pts, dtype=np.int32),
            'visibilities': np.asarray(visibilities, dtype=np.int32),
        }

        return annotations
