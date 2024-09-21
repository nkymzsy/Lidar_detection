#include "include/DetectNet.hpp"
#include "tools/KittiReader.hpp"
#include "tools/RosUtils.hpp"
#include "tools/TicToc.hpp"

#include <pcl/io/pcd_io.h>
int main(int argc, char **argv)
{
    ros::init(argc, argv, "demo");
    ros::NodeHandle nh;
    KittiDataReader kittiDataReader("/home/data/dataset/KITTIDetection/data_object_velodyne/training/velodyne/",
                                    "/home/data/dataset/KITTIDetection/training/label_2/",
                                    "/home/data/dataset/KITTIDetection/calib/");

    ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>("point_cloud_topic", 10);
    ros::Publisher boxPub = nh.advertise<visualization_msgs::MarkerArray>("bbox", 10);
    ros::Rate loopRate(0.5);

    int loop = 0;
    Detector denet(Detector::Mode::INFERENCE);
    denet.LoadModeParamters("/home/data/code/catkin_ws/src/pillar_detect/pt/model_20_loop.pt");
    std::vector<Object> objs;
    while (ros::ok())
    {
        auto &data = kittiDataReader.getOnceData();
        if (!data.dataIsOk)
        {
            break;
        }

        objs.clear();
        TicToc tic;
        denet.Infer(data.cloud, objs);
        tic.toc("whole infer:");
        rosutils::Publish3DBoundingBox(objs, boxPub);
        rosutils::PublishPointCloud(data.cloud, pub);
        loopRate.sleep();
    }

    return 0;
}