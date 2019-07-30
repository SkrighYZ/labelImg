import os
import numpy as np
import tensorflow as tf
import cv2 as cv
from lxml import etree as ET

def inference(modelpath, imgpath):
    # Read the graph.
    with tf.gfile.GFile(os.path.join(modelpath, 'frozen_inference_graph.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Session() as sess:
        # Restore session
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        # Read and preprocess an image.
        img = cv.imread(imgpath)
        rows = img.shape[0]
        cols = img.shape[1]
        inp = cv.resize(img, (224, 224))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

        # Run the model
        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                        sess.graph.get_tensor_by_name('detection_scores:0'),
                        sess.graph.get_tensor_by_name('detection_boxes:0'),
                        sess.graph.get_tensor_by_name('detection_classes:0')],
                        feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

        boxes = []
        labels = []
        num_detections = int(out[0][0])
        for i in range(num_detections):
            label = str(int(out[3][0][i]) - 1) # Only works in ECC case, may add labelmap later
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            if score > 0.6:
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                boxes.append([int(x), int(y), int(right), int(bottom)])
                labels.append(label)

    return boxes, labels, img.shape

def save_labelfile(imgpath, xmlpath, boxes, labels, shape):
    folder = os.path.basename(os.path.dirname(xmlpath))
    imgname = os.path.basename(imgpath)
    height, width, depth = shape
    obj_num = len(labels)

    root = ET.Element('annotation')
    ET.SubElement(root, 'folder').text = folder
    ET.SubElement(root, 'filename').text = imgname
    ET.SubElement(root, 'path').text = imgpath
    source = ET.SubElement(root, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'
    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(depth)
    ET.SubElement(root, 'segmented').text = '0'

    for i in range(0, obj_num):
        xmin, ymin, xmax, ymax = boxes[i]
        obj = ET.SubElement(root, 'object')
        ET.SubElement(obj, 'name').text = labels[i]
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(xmin)
        ET.SubElement(bndbox, 'ymin').text = str(ymin)
        ET.SubElement(bndbox, 'xmax').text = str(xmax)
        ET.SubElement(bndbox, 'ymax').text = str(ymax)
    tree = ET.ElementTree(root)
    tree.write(xmlpath, pretty_print=True)

if __name__ == '__main__':
    boxes, labels, shape = inference('test1.jpg')
    save_labelfile(os.path.join(os.getcwd(), 'test1.jpg'), 
            os.path.join(os.getcwd(), 'test1.xml'),
            boxes, labels, shape)

