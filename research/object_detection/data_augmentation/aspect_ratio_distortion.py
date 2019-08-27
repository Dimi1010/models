from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import warnings

import math
from random import uniform

import tensorflow as tf
from PIL import Image

from data_augmentation import augmentation_utils as aug_utils
from data_augmentation.augmentation_utils import ImageContainer, ObjectContainer

from utils import visualization_utils as VSU

flags = tf.app.flags
flags.DEFINE_string('image_dir', '', 'Path to the image directory')
flags.DEFINE_string('xml_dir', '', 'Path to the xml directory')
flags.DEFINE_string('output_dir', '', 'Path to the output directory')
flags.DEFINE_integer('runs_per_object', 1, 'How many runs should the script run for each object')

FLAGS = flags.FLAGS


def aspect_ratio_distortion(image_info: ImageContainer, image_path, output_dir):
    # Config loading
    runs_per_object = FLAGS.runs_per_object

    # Pre-processing
    image: Image.Image = Image.open(fp=image_path)

    # Object counter to be added to filename
    object_counter = 1
    for object in image_info.object_list:
        for run in range(0, runs_per_object):
            # Pre-processing image variables
            image_c = image.copy()
            image_info_c = copy.deepcopy(image_info)

            # Selecting multipliers
            x_ratio_multiplier: float = uniform(0.85, 1)
            y_ratio_multiplier: float = uniform(0.5, 1)

            # Updating size variables
            new_width = int(math.ceil(object.width * x_ratio_multiplier))
            new_height = int(math.ceil(object.height * y_ratio_multiplier))

            new_xmax = object.xmin + new_width
            new_ymax = object.ymin + new_height

            # Calculating unused space
            # Right rectangle
            right_rect = ObjectContainer(label='right_rect',
                                         xmin=new_xmax + 1,
                                         ymin=object.ymin,
                                         xmax=object.xmax,
                                         ymax=new_ymax + 1
                                         )
            # Bottom rectangle
            bottom_rect = ObjectContainer(label='bottom_rect',
                                          xmin=object.xmin,
                                          ymin=new_ymax + 1,
                                          xmax=object.xmax,
                                          ymax=object.ymax
                                          )
            # Get object and resize
            image_object = image_c.crop(box=object.get_bbox())
            image_object = image_object.resize(size=(new_width, new_height))

            # Get patches of background to replace the unused space with
            # Right rectangle generator
            right_generator = aug_utils.generate_background_patch(object=right_rect, image_info=image_info_c)

            # If generation fails cancels run
            if not right_generator[0]:
                continue
            # Bottom rectangle generator
            bottom_generator = aug_utils.generate_background_patch(object=bottom_rect, image_info=image_info_c)
            # If generation fails cancels run
            if not bottom_generator[0]:
                continue

            image_right_rect = aug_utils.get_patch_from_image_generator(image=image_c, generator=right_generator)
            image_bottom_rect = aug_utils.get_patch_from_image_generator(image=image_c, generator=bottom_generator)

            # Updating Image
            obj_paste_pos = (object.xmin, object.ymin)
            right_rect_pos = (right_rect.xmin, right_rect.ymin)
            bottom_rect_pos = (bottom_rect.xmin, bottom_rect.ymin)

            image_c.paste(image_object, box=obj_paste_pos)
            image_c.paste(image_right_rect, box=right_rect_pos)
            image_c.paste(image_bottom_rect, box=bottom_rect_pos)

            # Updating Image metadata
            object_c = image_info_c.find_matching_object(object=object)
            object_c.xmax = object_c.xmin + new_width
            object_c.ymax = object_c.ymin + new_height
            object_c.recalculate_size()

            # Saving
            pass

            # DEBUG
            import numpy as np

            npArr = np.array([[object_c.ymin / image_info_c.height, object_c.xmin / image_info_c.width, object_c.ymax / image_info_c.height, object_c.xmax / image_info_c.width],
                                 [right_rect.ymin / image_info_c.height, right_rect.xmin / image_info_c.width, right_rect.ymax / image_info_c.height, right_rect.xmax / image_info_c.width],
                                 [bottom_rect.ymin / image_info_c.height, bottom_rect.xmin / image_info_c.width, bottom_rect.ymax / image_info_c.height, bottom_rect.xmax / image_info_c.width]
                                 ])
            VSU.draw_bounding_boxes_on_image(image=image_c,
                                             boxes=npArr
                                             )
            image_c.show()
            print('Run:', run,'X mult:', x_ratio_multiplier, 'Y mult:', y_ratio_multiplier)
        object_counter += 1


def main(_):
    # Config loading
    img_dir = FLAGS.image_dir
    xml_dir = FLAGS.xml_dir
    out_dir = FLAGS.output_dir

    image_num = 0
    for file in os.listdir(xml_dir):
        if file.endswith('.xml'):
            xml_path = os.path.join(xml_dir, file)
            image_info = aug_utils.read_xml(xml_path)

            image_path = os.path.join(img_dir, image_info.filename)
            if not os.path.isfile(image_path):
                warnings.warn('Linked Image not found! Skipping...', UserWarning)
                continue

            print('Image No: ', image_num)
            image_num += 1
            aspect_ratio_distortion(image_info=image_info, image_path=image_path, output_dir=out_dir)


if __name__ == '__main__':
    tf.compat.v1.app.run(main=main)
