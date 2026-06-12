# -*- coding: utf-8 -*-
"""
Resilient image reading for shared and network backed storage

A single decode entry point tolerates transient input output faults common to clustered
filesystems, where a momentary stall returns an empty read rather than raising

The reader retries failed decodes with exponential backoff so a brief storage hiccup does not
escalate into a fatal crash that discards hours of training progress
"""

import logging
import time

import cv2

logger = logging.getLogger(__name__)


class ImageReadError(Exception):
    """
    Raised when an image cannot be decoded after the retry budget is exhausted

    The path and the attempt count are stored as optional attributes so a caller can log or
    branch on the failure programmatically rather than parsing the message
    """

    def __init__(self, message, *, img_path=None, attempts=None):
        super().__init__(message)
        self.img_path = img_path
        self.attempts = attempts


def read_image_rgb(img_path, retries=3, base_delay=0.5):
    """
    Decode an image file into an RGB array while tolerating transient read failures

    Retries on an empty decode using exponential backoff to absorb momentary storage stalls
    before giving up and surfacing the failure to the caller

    Args:
        img_path: Filesystem path to the image to decode
        retries: Maximum number of decode attempts before failing
        base_delay: Initial backoff delay in seconds doubled after each failed attempt

    Returns:
        The decoded image as an RGB ordered array

    Raises:
        ImageReadError: When the image cannot be decoded after all attempts
    """
    for attempt in range(retries):
        img = cv2.imread(img_path)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if attempt < retries - 1:
            logger.warning("Transient read failure %d/%d at %s", attempt + 1, retries, img_path)
            time.sleep(base_delay * (2**attempt))
    raise ImageReadError(
        f"Unreadable after {retries} attempts at {img_path}",
        img_path=img_path,
        attempts=retries,
    )
