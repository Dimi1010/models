"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('image_dir', '', 'Path to images')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
def class_text_to_int(row_label):
	if row_label == '03001210':
		return 1
	elif row_label == 'label_03001210':
		return 2
	elif row_label == '03001211':
		return 3
	elif row_label == 'label_03001211':
		return 4
	elif row_label == '03001220':
		return 5
	elif row_label == 'label_03001220':
		return 6
	elif row_label == '00062393':
		return 7
	elif row_label == '00062397':
		return 8
	elif row_label == '00062394':
		return 9
	elif row_label == '00062390':
		return 10
	elif row_label == 'label_00062393':
		return 11
	elif row_label == 'label_00062397':
		return 12
	elif row_label == 'label_00062394':
		return 13
	elif row_label == 'label_00021624':
		return 14
	elif row_label == '00021624':
		return 15
	elif row_label == 'label_00050887':
		return 16
	elif row_label == '00050887':
		return 17
	elif row_label == '00050886':
		return 18
	elif row_label == 'label_00050886':
		return 19
	elif row_label == 'label_00062390':
		return 20
	elif row_label == 'label_00062413':
		return 21
	elif row_label == 'label_00050965':
		return 22
	elif row_label == '00050965':
		return 23
	elif row_label == 'label_00062395':
		return 24
	elif row_label == 'label_00062412':
		return 25
	elif row_label == '00062412':
		return 26
	elif row_label == '00062413':
		return 27
	elif row_label == '00062395':
		return 28
	elif row_label == 'label_00062396':
		return 29
	elif row_label == '00062396':
		return 30
	elif row_label == 'label_00020162':
		return 31
	elif row_label == '00020162':
		return 32
	elif row_label == 'label_03100931':
		return 33
	elif row_label == '03100931':
		return 34
	elif row_label == 'label_00014557':
		return 35
	elif row_label == '00014557':
		return 36
	elif row_label == 'label_00014561':
		return 37
	elif row_label == '00014561':
		return 38
	elif row_label == 'label_00073597':
		return 39
	elif row_label == '00073597':
		return 40
	elif row_label == 'label_02802507':
		return 41
	elif row_label == '02802507':
		return 42
	elif row_label == 'label_00015905':
		return 43
	elif row_label == '00015905':
		return 44
	elif row_label == 'label_00015906':
		return 45
	elif row_label == '00015906':
		return 46
	elif row_label == 'label_00021105':
		return 47
	elif row_label == '00021105':
		return 48
	elif row_label == 'label_00066416':
		return 49
	elif row_label == '00066416':
		return 50
	elif row_label == 'label_00066417':
		return 51
	elif row_label == '00066417':
		return 52
	elif row_label == 'label_00062537':
		return 53
	elif row_label == '00062537':
		return 54
	elif row_label == 'label_00062538':
		return 55
	elif row_label == '00062538':
		return 56
	elif row_label == 'label_00062434':
		return 57
	elif row_label == '00062434':
		return 58
	elif row_label == 'label_00062447':
		return 59
	elif row_label == '00062447':
		return 60
	elif row_label == '00062541':
		return 61
	elif row_label == '00073709':
		return 62
	elif row_label == 'label_00073709':
		return 63
	elif row_label == 'label_00073675':
		return 64
	elif row_label == '00073675':
		return 65
	elif row_label == 'label_00015907':
		return 66
	elif row_label == '00015907':
		return 67
	elif row_label == 'label_02802511':
		return 68
	elif row_label == '02802511':
		return 69
	elif row_label == 'label_02802510':
		return 70
	elif row_label == '02802510':
		return 71
	elif row_label == 'label_00011325':
		return 72
	elif row_label == 'label_00062436':
		return 73
	elif row_label == '00062436':
		return 74
	elif row_label == 'label_00062435':
		return 75
	elif row_label == '00062435':
		return 76
	elif row_label == '00011325':
		return 77
	elif row_label == 'label_00054501':
		return 78
	elif row_label == '00054501':
		return 79
	elif row_label == 'label_00062541':
		return 80
	elif row_label == 'label_00049378':
		return 81
	elif row_label == '00049378':
		return 82
	elif row_label == 'label_00049294':
		return 83
	elif row_label == '00049294':
		return 84
	elif row_label == 'label_00049296':
		return 85
	elif row_label == '00049296':
		return 86
	elif row_label == 'label_00070844':
		return 87
	elif row_label == '00070844':
		return 88
	elif row_label == 'label_00053853':
		return 89
	elif row_label == '00053853':
		return 90
	elif row_label == 'label_00060668':
		return 91
	elif row_label == '00060668':
		return 92
	elif row_label == 'label_00060691':
		return 93
	elif row_label == '00060691':
		return 94
	elif row_label == '00057856':
		return 95
	elif row_label == 'label_00052557':
		return 96
	elif row_label == '00052557':
		return 97
	elif row_label == 'label_00052559':
		return 98
	elif row_label == '00052559':
		return 99
	elif row_label == 'label_00052560':
		return 100
	elif row_label == '00052560':
		return 101
	elif row_label == '00033379':
		return 102
	elif row_label == '00060885':
		return 103
	elif row_label == 'label_00033379':
		return 104
	elif row_label == 'label_00060885':
		return 105
	elif row_label == 'label_00032190':
		return 106
	elif row_label == '00032190':
		return 107
	elif row_label == 'label_00052556':
		return 108
	elif row_label == '00052556':
		return 109
	elif row_label == 'label_00052558':
		return 110
	elif row_label == '00052558':
		return 111
	elif row_label == 'label_00052555':
		return 112
	elif row_label == '00052555':
		return 113
	elif row_label == 'label_00030453':
		return 114
	elif row_label == '00030453':
		return 115
	elif row_label == 'label_00030448':
		return 116
	elif row_label == '00030448':
		return 117
	elif row_label == 'label_00030424':
		return 118
	elif row_label == '00030424':
		return 119
	elif row_label == 'label_00057855':
		return 120
	elif row_label == '00057855':
		return 121
	elif row_label == 'label_00070845':
		return 122
	elif row_label == '00070845':
		return 123
	elif row_label == 'label_00069581':
		return 124
	elif row_label == '00069581':
		return 125
	elif row_label == 'label_00069582':
		return 126
	elif row_label == '00069582':
		return 127
	elif row_label == 'label_00049295':
		return 128
	elif row_label == '00049295':
		return 129
	elif row_label == 'label_00057856':
		return 130
	elif row_label == 'label_00062559':
		return 131
	elif row_label == '00062559':
		return 132
	elif row_label == 'label_00050882':
		return 133
	elif row_label == '00050882':
		return 134
	else:
		None

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
