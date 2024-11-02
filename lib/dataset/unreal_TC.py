# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import json_tricks as json
from collections import OrderedDict
from copy import deepcopy

import numpy as np
from scipy.io import loadmat, savemat

from dataset.JointsDatasetTC import JointsDatasetTC


logger = logging.getLogger(__name__)


class UNRLTCDataset(JointsDatasetTC):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)

        self.num_joints = 16
        self.flip_pairs = []
        self.parent_ids = [0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13, 14, 15)
        self.lower_body_ids = (0, 1, 2, 3, 4, 5, 6)

        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        # create train/val split
        file_name = os.path.join(
            self.root, 'annot', self.image_set+'.json'
        )
        with open(file_name) as anno_file:
            anno = json.load(anno_file)

        gt_db = []
        for a in anno:
            image_name = a['image']

            c = np.array(a['center'], dtype=np.float)
            s = np.array([a['scale'], a['scale']], dtype=np.float)

            # Adjust center/scale slightly to avoid cropping limbs
            # <-- necessary?
            if c[0] != -1:
                c[1] = c[1] + 15 * s[1]
                s = s * 1.25

            # MPII uses matlab format, index is based 1,
            # we should first convert to 0-based index
            c = c - 1

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints,  3), dtype=np.float)
            if self.image_set != 'test':
                joints = np.array(a['joints'])
                joints[:, 0:2] = joints[:, 0:2] - 1
                joints_vis = np.array(a['joints_vis'])
                assert len(joints) == self.num_joints, \
                    'joint num diff: {} vs {}'.format(len(joints),
                                                      self.num_joints)

                joints_3d[:, 0:2] = joints[:, 0:2]
                # joints_3d[:, 2] = joints[:, 2]  # depth
                joints_3d_vis[:, 0] = joints_vis[:]
                joints_3d_vis[:, 1] = joints_vis[:]

            image_dir = 'images.zip@' if self.data_format == 'zip' else 'images'
            gt_db.append(
                {
                    'image': os.path.join(self.root, image_dir, image_name),
                    'center': c,
                    'scale': s,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0,
                }
            )

        return gt_db

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        # convert 0-based index to 1-based index
        preds = preds[:, :, 0:2] + 1.0

        if output_dir:
            pred_file = os.path.join(output_dir, 'pred.mat')
            savemat(pred_file, mdict={'preds': preds})

        if 'test' in cfg.DATASET.TEST_SET:
            return {'Null': 0.0}, 0.0

        # gt_file = os.path.join(cfg.DATASET.ROOT,
        #                        'annot',
        #                        'gt_{}.mat'.format(cfg.DATASET.TEST_SET))
        # gt_dict = loadmat(gt_file)
        # jnt_missing = gt_dict['jnt_missing'].reshape((self.num_joints,-1))
        # pos_gt_src = gt_dict['pos_gt_src'].reshape((self.num_joints,2,-1))
        # bbox_src = gt_dict['bbox_src'].reshape((2,2,-1))

        file_name = os.path.join(
            self.root, 'annot', cfg.DATASET.TEST_SET + '.json'
        )
        with open(file_name) as anno_file:
            anno = json.load(anno_file)

        len_anno = len(anno)
        jnt_visible = np.zeros((self.num_joints, len_anno))
        pos_gt_src = np.zeros((self.num_joints, 2, len_anno))
        scale = np.ones((self.num_joints, len_anno))
        for idx, a in enumerate(anno):
            jnt_visible[:, idx] = np.array(a['joints_vis'])
            pos_gt_src[:, :, idx] = np.array(a['joints'])[:,:2]
            scale[:, idx] = a['scale'] * 200.0 / 10.0

        pos_pred_src = np.transpose(preds, [1, 2, 0])

        # jnt_visible = 1 - jnt_missing
        uv_error = pos_pred_src - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=1)

        # SC_BIAS = 0.6
        # bboxsize = bbox_src[1, :, :] - bbox_src[0, :, :]
        # bboxsize = np.linalg.norm(bboxsize, axis=0)
        # bboxsize *= SC_BIAS

        # scale = np.multiply(bboxsize, np.ones((len(uv_err), 1)))
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)

        jnt_count = np.sum(jnt_visible, axis=1)

        # <-- for what?
        threshold = 0.5
        less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                          jnt_visible)
        jnt_count_ = deepcopy(jnt_count)
        jnt_count_[jnt_count == 0] = 1
        PCKh = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count_)

        # save
        rng = np.arange(0, 0.5+0.01, 0.01)
        pckAll = np.zeros((len(rng), self.num_joints))

        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                              jnt_visible)
            pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1),
                                     jnt_count_)

        PCKh = np.ma.array(PCKh, mask=False)
        # PCKh.mask[6:8] = True

        jnt_count = np.ma.array(jnt_count, mask=False)
        # jnt_count.mask[6:8] = True
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [
            ('Wrist', PCKh[0]),
            ('Thumb 0', PCKh[1]),
            ('Thumb 1', PCKh[6]),
            ('Thumb 2', PCKh[11]),
            ('Index 0', PCKh[2]),
            ('Index 1', PCKh[7]),
            ('Index 2', PCKh[12]),
            ('Middle 0', PCKh[3]),
            ('Middle 1', PCKh[8]),
            ('Middle 2', PCKh[13]),
            ('Ring 0', PCKh[4]),
            ('Ring 1', PCKh[9]),
            ('Ring 2', PCKh[14]),
            ('Pinky 0', PCKh[5]),
            ('Pinky 1', PCKh[10]),
            ('Pinky 2', PCKh[15]),
            ('Mean', np.sum(PCKh * jnt_ratio)),
            ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
        ]
        name_value = OrderedDict(name_value)

        return name_value, name_value['Mean']
