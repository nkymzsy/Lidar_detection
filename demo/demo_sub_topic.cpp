#include "lib/include/DetectNet.hpp"
#include "tools/KittiReader.hpp"
#include "tools/RosUtils.hpp"

class CloudCallback
{
private:
    Detector& denet_;
    ros::Publisher boxPub_;
    ros::Publisher cloudPub_;
    std::vector<Object> objs;
    pcl::PointCloud<pcl::PointXYZI> cloud;

public:
    CloudCallback(ros::NodeHandle &nh, Detector& denet): denet_(denet)
    {
        boxPub_ = nh.advertise<visualization_msgs::MarkerArray>("bbox", 10);
        cloudPub_ = nh.advertise<sensor_msgs::PointCloud2>("point_cloud_topic", 10);
        cloud.reserve(100000);
    }
    void operator()(const sensor_msgs::PointCloud2ConstPtr &cloud_msg)
    {
        objs.clear();
        cloud.clear();
        pcl::fromROSMsg(*cloud_msg, cloud);
        denet_.Infer(cloud, objs, 0.1);
        rosutils::Publish3DBoundingBox(objs, boxPub_);
        rosutils::PublishPointCloud(cloud, cloudPub_);
    }
};


int main(int argc, char **argv)
{
    ros::init(argc, argv, "demo");
    ros::NodeHandle nh;

    Detector denet(Detector::Mode::INFERENCE);
    denet.LoadModeParamters("/home/data/code/catkin_ws/temp/60epoches_model.pt");
    ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 10, CloudCallback(nh, denet));
    
    ros::spin();

    return 0;
}