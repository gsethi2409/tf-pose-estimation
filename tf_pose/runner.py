import base64
import os

import cv2
from functools import lru_cache

from tf_pose import common
from tf_pose import eval
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

Estimator = TfPoseEstimator


@lru_cache(maxsize=1)
def get_estimator(model='cmu', resize='0x0'):
    w, h = model_wh(resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))

    return e


def infer(image, model='cmu', resize='0x0', resize_out_ratio=4.0):
    """

    :param image:
    :param model:
    :param resize:
    :param resize_out_ratio:
    :return: coco_style_keypoints array
    """
    w, h = model_wh(resize)
    e = get_estimator(model, resize)

    # estimate human poses from a single image !
    image = common.read_imgfile(image, None, None)
    if image is None:
        raise Exception('Image can not be read, path=%s' % image)
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)
    image_h, image_w = image.shape[:2]

    if "TERM_PROGRAM" in os.environ and 'iTerm' in os.environ["TERM_PROGRAM"]:
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        image_str = cv2.imencode(".jpg", image)[1].tostring()
        print("\033]1337;File=name=;inline=1:" + base64.b64encode(image_str).decode("utf-8") + "\a")

    return [(eval.write_coco_json(human, image_w, image_h), human.score) for human in humans]
