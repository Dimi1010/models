from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import warnings

import tensorflow as tf
from PIL import Image

from data_augmentation import augmentation_utils as aug_utils
from data_augmentation.augmentation_utils import ImageContainer, ObjectContainer

flags = tf.app.flags
flags.DEFINE_string('image_dir', '', 'Path to the image directory')
flags.DEFINE_string('xml_dir', '', 'Path to the xml directory')
flags.DEFINE_string('output_dir', '', 'Path to the output directory')

FLAGS = flags.FLAGS


# Main processing method
def object_hiding(image_info: ImageContainer, image_path, output_dir):
    # Pre-processing
    image: Image.Image = Image.open(fp=image_path)

    # Object counter to be added to filename
    object_counter = 1
    for object in image_info.object_list:
        # Pre-processing for object specific image
        image_c = image.copy()
        image_info_c = copy.deepcopy(image_info)

        patch_generator = aug_utils.generate_background_patch(object=object, image_info=image_info_c)

        # Skips the object if box generation failed
        if not patch_generator[0]:
            continue

        # Gets patch from the box generator
        bbox_image = aug_utils.get_patch_from_image_generator(image=image_c, generator=patch_generator)

        paste_pos = (object.xmin, object.ymin)
        image_c.paste(bbox_image, box=paste_pos)

        # Post-processing image metadata
        image_info_c.remove_object(image_info_c.find_matching_object(object))

        filename = image_info_c.filename
        filename = os.path.splitext(filename)
        new_filename = filename[0] + '_H' + str(object_counter)
        new_filename = new_filename + filename[1]
        image_info_c.filename = new_filename

        # Saving data to output
        aug_utils.save(image_c, image_info_c, output_dir)

        # Post-processing variables
        object_counter += 1


def main(_):
    # Config Loading
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
            object_hiding(image_info=image_info, image_path=image_path, output_dir=out_dir)


if __name__ == '__main__':
    tf.compat.v1.app.run(main=main)
