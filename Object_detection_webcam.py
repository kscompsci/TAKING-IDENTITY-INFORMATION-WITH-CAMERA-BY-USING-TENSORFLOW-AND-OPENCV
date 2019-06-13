######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/20/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a webcam feed.
# It draws boxes and scores around the objects of interest in each frame from
# the webcam.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.


# Import packages
import os
import pymysql
import cv2
import numpy as np
import tensorflow as tf
import sys
import urllib
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
from pokedex import find_screen

# Name of the directory containing the object detection module we're using
def ProcessingFunction(image, flagForSteps):
    MODEL_NAME = 'inference_graph'

    # Grab path to current working directory
    CWD_PATH = os.getcwd()

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

    # Number of classes the object detector can identify
    NUM_CLASSES = 3

    ## Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `king`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)


    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    import matplotlib as plt
    def run_inference_for_single_image(image, graph):
        with graph.as_default():
        #with tf.Session() as sess:
          # Get handles to input and output tensors
          ops = tf.get_default_graph().get_operations()
          all_tensor_names = {output.name for op in ops for output in op.outputs}
          tensor_dict = {}
          for key in [
              'num_detections', 'detection_boxes', 'detection_scores',
              'detection_classes', 'detection_masks'
          ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
              tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                  tensor_name)
          if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
          image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

          # Run inference
          output_dict = sess.run(tensor_dict,
                                 feed_dict={image_tensor: np.expand_dims(image, 0)})

          # all outputs are float32 numpy arrays, so convert types as appropriate
          output_dict['num_detections'] = int(output_dict['num_detections'][0])
          output_dict['detection_classes'] = output_dict[
              'detection_classes'][0].astype(np.uint8)
          output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
          output_dict['detection_scores'] = output_dict['detection_scores'][0]
          if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict
    #url = "http://192.168.1.53:8080/shot.jpg"

    # Initialize webcam feed
    #video = cv2.VideoCapture(1)
    #if video.isOpened() :
    #    print("opened")
    #ret = video.set(3,1080)
    #ret = video.set(4,720)
    xy = 0
    ax=0
    temp = 0

    """imgResp = urllib.request.urlopen(url)
    
    # Numpy to convert into a arrayqq
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    
    # Finally decode the array to OpenCV usable format ;)
    
    frame = cv2.imdecode(imgNp, -1)"""
    frame = image;
    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    #ret, frame = video.read(0)
    #frame = cv2.flip(frame, 1)

    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})


    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.50)
    # Draw the results of the detection (aka 'visulaize the results')
    output_dict = run_inference_for_single_image(frame, detection_graph)
    max_boxes_to_draw = output_dict['detection_boxes'].shape[0]
    for i in range(min(max_boxes_to_draw, output_dict['detection_boxes'].shape[0])):
        if output_dict['detection_scores'][i] > 0.80:
            if output_dict['detection_classes'][i] in category_index.keys():
                class_name = category_index[output_dict['detection_classes'][i]]['name']
                #print(output_dict['detection_boxes'][i])
                ymin = boxes[0, i, 0]
                xmin = boxes[0, i, 1]
                ymax = boxes[0, i, 2]
                xmax = boxes[0, i, 3]
               # im_width = 1920
                #im_height = 1080
                im_height, im_width = image.shape[:2]
                (xminn, xmaxx, yminn, ymaxx) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
                #crop_img=tf.image.crop_to_bounding_box(frame,int(yminn), int(xminn), int(ymaxx-yminn), int(xmaxx-xminn))

                if(output_dict['detection_classes'][i] !=1 ):
                    if xminn-70>=0:
                        xminn= xminn-70
                    if xmaxx+70<im_width:
                        xmaxx= xmaxx+70
                    if yminn-70>=0:
                        yminn= yminn-70
                    if ymaxx+70<=im_height:
                        ymaxx= ymaxx+70
                temp = output_dict['detection_classes'][i]
                crop_img=frame[int(yminn):int(ymaxx),int(xminn):int(xmaxx)]

                # print(session.run(file))

                """crop_img = frame[int((output_dict['detection_boxes'][i][0]) * 720): int(
                    (output_dict['detection_boxes'][i][2]) * 720),
                           int((output_dict['detection_boxes'][i][1]) * 1080):int(
                               (output_dict['detection_boxes'][i][3]) * 1080)]"""
                if(flagForSteps==1):
                    cv2.imshow("Cropped by Tensorflow",crop_img)
                #print(class_name)
        if (flagForSteps == 1):
           cv2.imshow('Object detector', frame)
           cv2.waitKey(0)

    # Press 'q' to quit

    try:
        tcnum = find_screen.pokedex_find_screen(crop_img, temp, flagForSteps)

    except:

        return "crop_img error"

    #tcnum="crop img error"
    # Clean up
    #print("dfjsdfkjsdf"+tcnum)
    #video.release()
    cv2.destroyAllWindows()
    return tcnum


