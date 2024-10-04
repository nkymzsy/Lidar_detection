#include "lib/include/DetectNet.hpp"
#include "tools/KittiReader.hpp"
#include "tools/RosUtils.hpp"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "demo");
    ros::NodeHandle nh;
    std::string cloud_path = "/home/data/dataset/KITTIDetection/data_object_velodyne/training/velodyne/";
    std::string label_path = "/home/data/dataset/KITTIDetection/training/label_2/";
    std::string calib_path = "/home/data/dataset/KITTIDetection/calib/";
    KittiDataReader kittiDataReader(cloud_path, label_path, calib_path);

    ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>("point_cloud_topic", 10);
    ros::Publisher boxPub = nh.advertise<visualization_msgs::MarkerArray>("bbox", 10);
    ros::Publisher boxRealPub = nh.advertise<visualization_msgs::MarkerArray>("real_bbox", 10);
    ros::Rate loopRate(0.5);

    Detector denet(Detector::Mode::INFERENCE);
    denet.LoadModeParamters("/home/data/code/catkin_ws/temp/60epoches_model.pt");
    std::vector<Object> objs;
    while (ros::ok())
    {
        auto data = kittiDataReader.getOnceData();
        if (data == nullptr)
        {
            break;
        }
        objs.clear();
        auto &[cloud, objects] = *data;
        TicToc tic;
        denet.Infer(cloud, objs, 0.1);
        tic.toc("whole infer:");
        rosutils::Publish3DBoundingBox(objs, boxPub);
        rosutils::Publish3DBoundingBox(data->second, boxRealPub, 0.5);
        rosutils::PublishPointCloud(cloud, pub);
        loopRate.sleep();
    }

    return 0;
}