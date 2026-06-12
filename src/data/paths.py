# -*- coding: utf-8 -*-
"""
Dataset path globals derived from the DATASET environment variable

This module is the single owner of .env loading, the file is read from the repo root so imports
resolve identically from any working directory, DATASET points at the shared partition and COCO
lives under DATASET/coco, every loader shares these locations as a single source of truth

IMPORTANT: the class selection is multilabel and lives in config/coco.yaml, these paths are
class agnostic and do not change when the enabled classes change
"""

import os

from dotenv import load_dotenv

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv(os.path.join(BASE_PATH, ".env"))  # Resolved against the repo root so the launch directory never matters

DATASET = os.environ.get("DATASET")  # Shared partition root
if not DATASET:
    raise RuntimeError("DATASET is not set, copy .env.example to .env and point it at the data partition")

COCO_ROOT = os.path.join(DATASET, "coco")  # COCO lives directly under the partition root

DATA_TRAIN = os.path.join(COCO_ROOT, "train2017")  # Labelled images
DATA_UNLABELED = os.path.join(COCO_ROOT, "unlabeled2017")  # Unlabelled images
DATA_VAL = os.path.join(COCO_ROOT, "val2017")  # Validation images
DATA_TRAIN_ANNOTATIONS = os.path.join(COCO_ROOT, "annotations", "instances_train2017.json")
DATA_VAL_ANNOTATIONS = os.path.join(COCO_ROOT, "annotations", "instances_val2017.json")
