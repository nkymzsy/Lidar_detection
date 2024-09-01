

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/tf.h>

#include "tools/KittiReader.hpp"
#include "include/DetectNet.hpp"

void publishPointCloud(pcl::PointCloud<pcl::PointXYZI> &cloud, ros::Publisher &pub)
{
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(cloud, cloud_msg);
    cloud_msg.header.frame_id = "world"; 
    cloud_msg.header.stamp = ros::Time::now();
    pub.publish(cloud_msg);
}

tf::Quaternion computeQuaternionFromHeading(float heading)
{
    tf::Quaternion q;
    q.setRPY(0, 0, heading); // 绕 z 轴旋转
    return q;
}

void publish3DBoundingBox(std::vector<Object> &objects, ros::Publisher &marker_pub)
{
    visualization_msgs::MarkerArray empty_markers;
    visualization_msgs::Marker delete_all_marker;
    delete_all_marker.action = visualization_msgs::Marker::DELETEALL;
    empty_markers.markers.push_back(delete_all_marker);
    marker_pub.publish(empty_markers);

    int i = 0;
    visualization_msgs::MarkerArray markers_array;
    for (const auto &object : objects)
    {
        visualization_msgs::Marker marker;
        marker.header.frame_id =  "world";
        marker.header.stamp = ros::Time::now();
        marker.ns = "3d_boxes";
        marker.id = i++;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;
        marker.lifetime = ros::Duration(0);

        // 设置颜色
        marker.color.r = 1.0f;
        marker.color.g = 1.0f;
        marker.color.b = 0.0f;
        marker.color.a = 0.7f; // 半透明

        // 设置尺寸
        marker.scale.x = object.dimensions[0]; // 长度
        marker.scale.y = object.dimensions[1]; // 宽度
        marker.scale.z = object.dimensions[2]; // 高度

        // 设置位置
        geometry_msgs::Point center;
        center.x = object.position[0];
        center.y = object.position[1];
        center.z = object.position[2] + object.dimensions[2] / 2;
        marker.pose.position = center;

        // 设置方向
        tf::Quaternion quaternion = computeQuaternionFromHeading(-object.heading); 
        geometry_msgs::Quaternion orientation;
        tf::quaternionTFToMsg(quaternion, orientation);
        marker.pose.orientation = orientation;

        // 添加 Marker 到 MarkerArray
        markers_array.markers.push_back(marker);
    }
    marker_pub.publish(markers_array);
}
int main(int argc, char **argv)
{
    ros::init(argc, argv, "demo");
    ros::NodeHandle nh;
    KittiDataReader kittiDataReader("/home/data/dataset/KITTIDetection/data_object_velodyne/training/velodyne/",
                                    "/home/data/dataset/KITTIDetection/training/label_2/",
                                    "/home/data/dataset/KITTIDetection/calib/");

    ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>("point_cloud_topic", 10);
    ros::Publisher box_pub = nh.advertise<visualization_msgs::MarkerArray>("bbox", 10);
    ros::Rate loop_rate(10);

    Detector denet;

    auto &data = kittiDataReader.getOnceData();
     data = kittiDataReader.getOnceData();
     data = kittiDataReader.getOnceData();
     data = kittiDataReader.getOnceData();
    while (ros::ok())
    {
        ros::spinOnce();

        publishPointCloud(data.cloud, pub);
        publish3DBoundingBox(data.objects, box_pub);
        denet.train(data.cloud, data.objects);
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}