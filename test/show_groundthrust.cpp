
#include "lib/include/DetectNet.hpp"
#include "tools/KittiReader.hpp"
#include "tools/RosUtils.hpp"

// 本文件的作用是检测ground_truth中的heatmap，生成是否正确

int main(int argc, char **argv)
{
    ros::init(argc, argv, "demo");
    ros::NodeHandle nh;
    std::string cloud_path = "/home/data/dataset/KITTIDetection/data_object_velodyne/training/velodyne/";
    std::string label_path = "/home/data/dataset/KITTIDetection/training/label_2/";
    std::string calib_path = "/home/data/dataset/KITTIDetection/calib/";
    KittiDataReader kittiDataReader(cloud_path, label_path, calib_path);

    ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>("heatmap", 10);
    ros::Publisher boxPub = nh.advertise<visualization_msgs::MarkerArray>("bbox", 10);
    ros::Publisher boxRealPub = nh.advertise<visualization_msgs::MarkerArray>("real_bbox", 10);
    ros::Rate loopRate(0.5);

    Detector denet(Detector::Mode::INFERENCE);
    denet.LoadModeParamters("/home/data/code/catkin_ws/temp/60epoches_model.pt");
    std::vector<Object> objs;
    float heading = 0;
    float dim_x = 0.5;
    for (int i = -20; i < 30; i += 5)
    {
        objs.emplace_back(Object(0, heading, {i + 21, i, 1}, {dim_x, 2, 1}));
        heading += 0.25; 
        dim_x += 0.7;
    }

    Detector::TensorMap ground_truth;
    denet.BuildDetectionGroundTruth(objs, ground_truth);
    auto heatmap = ground_truth["heatmap"];
    pcl::PointCloud<pcl::PointXYZI> cloud;
    pcl::PointXYZI point;

    for (int i = 0; i < Config::bev_w; i++)
    {
        for (int j = 0; j < Config::bev_h; j++)
        {
            point.x = i * Config::pillar_x_size + Config::roi_x_min;
            point.y = j * Config::pillar_y_size + Config::roi_y_min;
            point.z = 1;
            point.intensity = heatmap.index({0, 0, i, j}).item<float>();
            cloud.push_back(point);
        }
    }

    while (ros::ok())
    {
        rosutils::Publish3DBoundingBox(objs, boxPub);
        rosutils::PublishPointCloud(cloud, pub);
        loopRate.sleep();
    }

    return 0;
}