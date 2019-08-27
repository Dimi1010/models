from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import warnings

import pandas as pd
import tensorflow as tf
from PIL import Image
from object_detection.utils import visualization_utils as vs

flags = tf.app.flags
flags.DEFINE_string('tfrecord_path', '', 'Path to the TF Record')

FLAGS = flags.FLAGS


class ObjectRecord:
    def __init__(self, filename, image, object_df):
        self.filename = filename
        self.image = image
        self.image_width, self.image_height = image.size
        self.object_df = object_df


def decodeExample(string_record):
    example = tf.train.Example()
    example.ParseFromString(string_record)

    height = example.features.feature['image/height'].int64_list.value[0]
    width = example.features.feature['image/width'].int64_list.value[0]

    filename = example.features.feature['image/filename'].bytes_list.value[0].decode("utf-8")
    source_id = example.features.feature['image/source_id'].bytes_list.value[0].decode("utf-8")
    encoded = example.features.feature['image/encoded'].bytes_list.value[0]
    format = example.features.feature['image/format'].bytes_list.value[0].decode("utf-8")

    xmin_list = list(example.features.feature['image/object/bbox/xmin'].float_list.value)
    xmax_list = list(example.features.feature['image/object/bbox/xmax'].float_list.value)
    ymin_list = list(example.features.feature['image/object/bbox/ymin'].float_list.value)
    ymax_list = list(example.features.feature['image/object/bbox/ymax'].float_list.value)

    text_list = list(example.features.feature['image/object/class/text'].bytes_list.value)
    text_list = [x.decode('UTF8') for x in text_list]

    label_list = list(example.features.feature['image/object/class/label'].int64_list.value)

    object_df = pd.DataFrame(
        {'class': text_list,
         'label': label_list,
         'xmin': xmin_list,
         'xmax': xmax_list,
         'ymin': ymin_list,
         'ymax': ymax_list
         }
    )

    encoded_io = io.BytesIO(encoded)
    image = Image.open(encoded_io)

    img_width, img_height = image.size
    if width != img_width or height != img_height:
        warnings.warn('Image width or height not corresponding to recorded width or height', UserWarning)
        answer = None
        while answer not in ('y', 'n', 'yes', 'no'):
            answer = input('Should the process skip this image? (y/n)')
            if answer in ('y', 'yes'):
                return False, None
            elif answer not in ('n', 'no'):
                print('Answer should one of these ', ['y', 'n', 'yes', 'no'])

    record: ObjectRecord = ObjectRecord(filename, image, object_df)
    return True, record


def processImage(record):
    object_df = record.object_df
    object_coords = object_df[['ymin', 'xmin', 'ymax', 'xmax']].as_matrix()
    object_labels = object_df['class'].tolist()

    image = record.image
    vs.draw_bounding_boxes_on_image(image=image, boxes=object_coords, display_str_list_list=object_labels)
    image.show()


def main(argv):
    tfrecord_path = FLAGS.tfrecord_path
    for string_record in tf.python_io.tf_record_iterator(tfrecord_path):
        valid, record = decodeExample(string_record)
        # Skips if record is invalid
        if not valid:
            continue
        processImage(record)

        input("Press Enter to continue to next image...")


if __name__ == '__main__':
    tf.app.run()
