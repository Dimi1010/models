import os
import math
import warnings

from random import randint
from typing import List
from typing import Tuple


from PIL import Image

from xml.dom import minidom
from xml.etree import ElementTree as ET


# Object container to store an object's information
class ObjectContainer:
    def __init__(self, label='NoLabelProvided', xmin: int = None, ymin: int = None, xmax: int = None, ymax: int = None):
        self.label = label
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.width = xmax - xmin
        self.height = ymax - ymin

    def __eq__(self, other):
        if not isinstance(other, ObjectContainer):
            return NotImplemented

        return (
            self.label == other.label and
            self.xmin == other.xmin and
            self.ymin == other.ymin and
            self.xmax == other.xmax and
            self.ymax == other.ymax
        )

    def get_bbox(self, mode='xy'):
        if mode == 'xy':
            return self.xmin, self.ymin, self.xmax, self.ymax
        elif mode == 'yx':
            return self.ymin, self.xmin, self.ymax, self.xmax
        else:
            raise ValueError("mode not from selection ['xy' , 'yx']")

    def recalculate_size(self):
        self.width = self.xmax - self.xmin
        self.height = self.ymax - self.ymin

    def do_overlap(self, xmin, ymin, xmax, ymax) -> bool:
        if xmin is None or ymin is None or xmax is None or ymax is None:
            raise TypeError("Input values contain 'None'")

        # One rectangle is to the left of the other
        if self.xmin > xmax or xmin > self.xmax:
            return False

        # One rectangle is above the other
        if self.ymin > ymax or ymin > self.ymax:
            return False

        # Rectangles overlap
        return True


# Image container to store an image's information
class ImageContainer:
    def __init__(self, filename='NoNameProvided', width=0, height=0, depth=0):
        self.filename = filename
        self.width = width
        self.height = height
        self.depth = depth
        self.object_list: List[ObjectContainer] = []

    def append_object(self, object: ObjectContainer):
        if object is None:
            raise TypeError('Parsed object is of type None')
        else:
            self.object_list.append(object)

    def remove_object(self, object: ObjectContainer):
        if object is None:
            raise TypeError('Parsed object is of type None')
        else:
            self.object_list.remove(object)

    def find_matching_object(self, object: ObjectContainer):
        for member in self.object_list:
            if object == member:
                return member
        return None

    def do_overlap_any(self, xmin, ymin, xmax, ymax) -> (bool, ObjectContainer):
        for object in self.object_list:
            if object.do_overlap(xmin, ymin, xmax, ymax):
                return True, object
        return False, None


# Reads the XML
# Returns ImageContainer
def read_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find('filename').text

    size = root.find('size')
    width = int(size[0].text)
    height = int(size[1].text)
    depth = int(size[2].text)

    image_container = ImageContainer(filename=filename, width=width, height=height, depth=depth)

    for member in root.findall('object'):
        label = member.find('name').text
        bndbox = member.find('bndbox')

        xmin = int(bndbox[0].text)
        ymin = int(bndbox[1].text)
        xmax = int(bndbox[2].text)
        ymax = int(bndbox[3].text)

        image_container.append_object(ObjectContainer(label=label, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax))

    return image_container


# Saves the processed image to jpg and xml file
def save(image: Image, image_info: ImageContainer, output_dir):
    # Error checks
    if os.path.exists(output_dir) and not os.path.isdir(output_dir) or not output_dir:
        raise ValueError('output_dir is not a valid directory')

    # Preparing subdirectories
    img_dir = os.path.join(output_dir, 'images')
    xml_dir = os.path.join(output_dir, 'xml')

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(xml_dir, exist_ok=True)

    # Preparing XML metadata
    root = ET.Element('annotation')

    filename = ET.SubElement(root, 'filename')
    filename.text = image_info.filename

    size = ET.SubElement(root, 'size')

    width = ET.SubElement(size, 'width')
    width.text = str(image_info.width)

    height = ET.SubElement(size, 'height')
    height.text = str(image_info.height)

    depth = ET.SubElement(size, 'depth')
    depth.text = str(image_info.depth)

    # Writing object information
    for member in image_info.object_list:
        object = ET.SubElement(root, 'object')

        name = ET.SubElement(object, 'name')
        name.text = member.label

        bbox = ET.SubElement(object, 'bndbox')

        xmin = ET.SubElement(bbox, 'xmin')
        xmin.text = str(member.xmin)

        ymin = ET.SubElement(bbox, 'ymin')
        ymin.text = str(member.ymin)

        xmax = ET.SubElement(bbox, 'xmax')
        xmax.text = str(member.xmax)

        ymax = ET.SubElement(bbox, 'ymax')
        ymax.text = str(member.ymax)

    # Saving XML
    filename = os.path.splitext(image_info.filename)
    filename = filename[0] + '.xml'

    xml_path = os.path.join(xml_dir, filename)
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    with open(xml_path, 'w') as f:
        f.write(xml_str)

    # Saving Image
    image.save(fp=os.path.join(img_dir, image_info.filename))


# Returns a patch from the background of the image linked to the image container
# with size equal to the object container
def generate_background_patch(object: ObjectContainer, image_info: ImageContainer):
    # Pre-processing variables
    gen = 0
    split_mode = 0
    transposed = False
    generation_successful = False
    object_qrt = None

    # Box Selection Loop
    while True:
        gen += 1

        if gen < 10000:
            # Generates box on 1 to 1 scale
            bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = generate_bbox(object, image_info)
        elif gen < 20000:
            # Generates box on 1 to 1 scale (transposed)
            transposed = True
            bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = generate_bbox(object, image_info,
                                                                                 mode='transposed')
        elif gen < 30000:
            # Generates box with 1/4 of the area of the original
            split_mode = 1

            new_xmax = int(math.ceil((object.xmax - object.xmin) / 2 + object.xmin))
            new_ymax = int(math.ceil((object.ymax - object.ymin) / 2 + object.ymin))

            object_qrt = ObjectContainer(xmin=object.xmin,
                                         ymin=object.ymin,
                                         xmax=new_xmax,
                                         ymax=new_ymax
                                         )
            transposed = False
            bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = generate_bbox(object_qrt, image_info,
                                                                                 mode='normal')
        else:
            # Creates object_qrt if it hasn't been created
            if object_qrt is None:
                split_mode = 1

                new_xmax = int(math.ceil((object.xmax - object.xmin) / 2 + object.xmin))
                new_ymax = int(math.ceil((object.ymax - object.ymin) / 2 + object.ymin))

                object_qrt = ObjectContainer(xmin=object.xmin,
                                             ymin=object.ymin,
                                             xmax=new_xmax,
                                             ymax=new_ymax
                                             )

            # Generates box with 1/4 of the area of the original (transposed)
            transposed = True
            bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = generate_bbox(object_qrt, image_info,
                                                                                 mode='transposed')

        # Breaks if a non-overlapping container is found
        if not image_info.do_overlap_any(bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax)[0]:
            generation_successful = True
            break

        # Contingency against infinite loop
        if gen > 40000:
            break

    # Post-processing return variables
    bbox = [bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax]
    return generation_successful, bbox, transposed, split_mode


# Wrapper function for get_patch_from_image that accepts only image and tuple from generator
def get_patch_from_image_generator(image: Image.Image, generator: Tuple[bool, List[int], bool, int]) -> [Image.Image, None]:
    # Generation unsuccessful
    if not generator[0]:
        warnings.warn("Function was fed unsuccessful generator. Returning 'None' type.")
        return None
    return get_patch_from_image(image=image,
                                box=generator[1],
                                transposed=generator[2],
                                split_mode=generator[3]
                                )


# Returns a patch from an image from provided arguments
def get_patch_from_image(image: Image.Image, box: List[int], transposed: bool = False, split_mode: int = 0):
    image_patch = image.crop(box=box)

    if split_mode == 1:
        new_size = tuple(i * 2 for i in image_patch.size)
        image_patch = image_patch.resize(size=new_size)

    if transposed:
        image_patch = image_patch.transpose(method=Image.TRANSPOSE)
    return image_patch

# Generates bounding box for area of the image to replace the object with
def generate_bbox(object: ObjectContainer, image_info: ImageContainer, mode: str = 'normal'):
    if mode == 'normal':
        bound_right = image_info.width - object.width
        bound_bottom = image_info.height - object.height

        left = randint(0, bound_right)
        top = randint(0, bound_bottom)
        right = left + object.width
        bottom = top + object.height
    elif mode == 'transposed':
        bound_right = image_info.width - object.height
        bound_bottom = image_info.height - object.width

        left = randint(0, bound_right)
        top = randint(0, bound_bottom)
        right = left + object.height
        bottom = top + object.width
    else:
        raise ValueError("Mode not from specified list: 'normal', 'transposed'")

    return left, top, right, bottom
