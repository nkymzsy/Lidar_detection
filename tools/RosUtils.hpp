#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/tf.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include "../include/Datatype.hpp"

namespace rosutils
{
    inline void PublishPointCloud(pcl::PointCloud<pcl::PointXYZI> &cloud, ros::Publisher &pub)
    {
        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(cloud, cloud_msg);
        cloud_msg.header.frame_id = "world";
        cloud_msg.header.stamp = ros::Time::now();
        pub.publish(cloud_msg);
    }

    inline tf::Quaternion ComputeQuaternionFromHeading(float heading)
    {
        tf::Quaternion q;
        q.setRPY(0, 0, heading); // 绕 z 轴旋转
        return q;
    }

    inline void Publish3DBoundingBox(std::vector<Object> &objects, ros::Publisher &marker_pub)
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
            marker.header.frame_id = "world";
            marker.header.stamp = ros::Time::now();
            marker.ns = "3d_boxes";
            marker.id = i++;
            marker.type = visualization_msgs::Marker::CUBE;
            marker.action = visualization_msgs::Marker::ADD;
            marker.lifetime = ros::Duration(0);

            // 设置颜色
            if (object.label == 0)
            {
                marker.color.r = 1.0f;
                marker.color.g = 1.0f;
                marker.color.b = 0.0f;
                marker.color.a = 0.6f; // 半透明
            }
            else
            {
                marker.color.r = 0.0f;
                marker.color.g = 1.0f;
                marker.color.b = 0.0f;
                marker.color.a = 0.6f;
            }

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
            tf::Quaternion quaternion = ComputeQuaternionFromHeading(-object.heading);
            geometry_msgs::Quaternion orientation;
            tf::quaternionTFToMsg(quaternion, orientation);
            marker.pose.orientation = orientation;

            // 添加 Marker 到 MarkerArray
            markers_array.markers.push_back(marker);
        }
        marker_pub.publish(markers_array);
    }
}