#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from pr2_robot.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from pr2_robot.msg import DetectedObjectsArray
from pr2_robot.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    pcl_data = ros_to_pcl(pcl_msg)
    
    # TODO: Statistical Outlier Filtering
    outlier_filter = pcl_data.make_statistical_outlier_filter()

    # number of neighboring points to analyze for each point
    outlier_filter.set_mean_k(50)

    # threshold scale, adjusted for more aggressive filtering
    x = 0.1
    outlier_filter.set_std_dev_mul_thresh(x)
    outlier_filter.set_negative(False)

    pcl_filtered = outlier_filter.filter()
    
    # TODO: Voxel Grid Downsampling
    vox = pcl_filtered.make_voxel_grid_filter()
    LEAF_SIZE = 0.0035
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    pcl_filtered = vox.filter()

    # TODO: PassThrough Filter
    passthrough_Z = pcl_filtered.make_passthrough_filter()
    filter_axis = 'z'
    passthrough_Z.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 1.0
    passthrough_Z.set_filter_limits(axis_min, axis_max)
    pcl_filtered = passthrough_Z.filter()

    # Filter along X axis to remove bins, leaving only what is on the table
    passthrough_X = pcl_filtered.make_passthrough_filter()
    filter_axis = 'x'
    passthrough_X.set_filter_field_name(filter_axis)
    axis_min = .3
    axis_max = 1.1
    passthrough_X.set_filter_limits(axis_min, axis_max)
    pcl_filtered = passthrough_X.filter()


    # TODO: RANSAC Plane Segmentation
    seg = pcl_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    # TODO: Extract inliers and outliers
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)

    inliers, coefficients = seg.segment()

    pcl_table = pcl_filtered.extract(inliers, negative=False)
    pcl_objects = pcl_filtered.extract(inliers, negative=True)

    # TODO: Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(pcl_objects)
    tree = white_cloud.make_kdtree()

    ec = white_cloud.make_EuclideanClusterExtraction()
    # tolerances for distance threshold (TWEAK THESE)
    ec.set_ClusterTolerance(.01)
    ec.set_MinClusterSize(100)
    ec.set_MaxClusterSize(5000)

    # search k-d tree for clusters
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately

    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, index in enumerate(indices):
            color_cluster_point_list.append([white_cloud[index][0],
                                             white_cloud[index][1],
                                             white_cloud[index][2],
                                             rgb_to_float(cluster_color[j])])

    cloud_objects = pcl.PointCloud_PointXYZRGB()
    cloud_objects.from_list(color_cluster_point_list)

    # TODO: Convert PCL data to ROS messages

    ros_table = pcl_to_ros(pcl_table)
    ros_objects = pcl_to_ros(pcl_objects)
    ros_clusters = pcl_to_ros(cloud_objects)

    # TODO: Publish ROS messages
    pcl_objects_pub.publish(ros_objects)
    pcl_table_pub.publish(ros_table)
    pcl_cluster_pub.publish(ros_clusters)

# Exercise-3 TODOs:

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_cluster = pcl_objects.extract(pts_list)
        ros_cluster = pcl_to_ros(pcl_cluster)
        # Compute the associated feature vector

        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1, -1)))
        label = encoder.inverse_transform(prediction)[0]

        detected_objects_labels.append(label)

        # Publish a label into RViz

        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += 0.4
        object_markers_pub.publish(make_label(label, label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects, pcl_table)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list, pcl_table):

    # TODO: Initialize variables
    test_scene_num = Int32()
    test_scene_num.data = 2 # is there a way to determine this automatically?
    object_name = String()
    arm_name = String()
    pick_pose = Pose()
    place_pose = Pose()

    yaml_list = []

    # build a dictionary from object labels to centroids
    centroids = {}
    for object in object_list:
        label = object.label
        points_arr = ros_to_pcl(object.cloud).to_array()
        centroids[label] = np.mean(points_arr, axis=0)[:3]
        centroids[label] = [np.asscalar(c) for c in centroids[label]]

    # TODO: Get/Read parameters

    object_list_param = rospy.get_param('/object_list')
    dropbox_param = rospy.get_param('/dropbox')

    # TODO: Parse parameters into individual variables

    left_box_pos = None
    right_box_pos = None
    for box in dropbox_param:
        if box['name'] == 'left':
            left_box_pos = box['position']
        elif box['name'] == 'right':
            right_box_pos = box['position']

    # TODO:Rotate PR2 in place to capture side tables for the collision map

    # TODO: Loop through the pick list

    for object in object_list_param:

        object_name_param = object['name']
        object_group_param = object['group']

        object_name.data = object_name_param

        arm_name.data = 'left' if object_group_param == 'red' else 'right'
        
        collision_cloud = pcl.PointCloud_PointXYZRGB()
        
        collision_point_list = [p for p in pcl_table]
        
        # TODO: fix up collision so that previous objects don't stick around
        # current_obj_idx = None
        # for idx, obj in enumerate(object_list):
        #     if obj.label != object_name_param:
        #         pcl_obj = ros_to_pcl(obj.cloud)
        #         for pt in pcl_obj:
        #             collision_point_list.append(pt)
        #     else:
        #         current_obj_idx = idx

        # del object_list[current_obj_idx]

        collision_cloud.from_list(collision_point_list)
        ros_collision_cloud = pcl_to_ros(collision_cloud)
        
        collision_map_pub.publish(ros_collision_cloud)

        # TODO: Get the PointCloud for a given object and obtain it's centroid
        if object_name_param in centroids:
            centroid = centroids[object_name_param]
        else:
            print("Failed to identify {} in detected objects list".format(
                object_name_param))
            continue

        # pick_pose.position = Point()
        pick_pose.position.x = centroid[0]
        pick_pose.position.y = centroid[1]
        pick_pose.position.z = centroid[2]

        # TODO: Create 'place_pose' for the object
        target_box = left_box_pos if arm_name.data == 'left' else right_box_pos
        
        # place_pose.position = Point()
        place_pose.position.x = target_box[0]
        place_pose.position.y = target_box[1]
        place_pose.position.z = target_box[2]

        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        yaml_list.append(make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose))

        # NB: Skipping the actual pick and place routing due to limitations in the
        # configuration of the simulator
        
        # Wait for 'pick_place_routine' service to come up
        # rospy.wait_for_service('pick_place_routine')

        # try:
        #     pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

        #     # TODO: Insert your message variables to be sent as a service request
        #     resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)

        #     print ("Response: ",resp.success)

        # except rospy.ServiceException, e:
        #     print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file
    send_to_yaml('output_{}.yaml'.format(test_scene_num.data), yaml_list)

if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points",
                               pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_markers", DetectedObjectsArray, queue_size=1)
    collision_map_pub = rospy.Publisher("/pr2/3d_map/points", PointCloud2, queue_size=1)

    # TODO: Load Model From disk
    model = pickle.load(open('./model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes = model['classes']
    y_train = [c for l in [[x] * 50 for x in encoder.classes] for c in l]
    print(encoder.classes)
    encoder.fit_transform(y_train)
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
