#include <pcl/io/pcd_io.h>
#include "include/DetectNet.hpp"
#include "tools/KittiReader.hpp"
#include "tools/RosUtils.hpp"
int main(int argc, char **argv)
{
    ros::init(argc, argv, "demo");
    ros::NodeHandle nh;
    ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>("point_cloud_topic", 10);
    ros::Publisher box_pub = nh.advertise<visualization_msgs::MarkerArray>("bbox", 10);
    ros::Rate loop_rate(10);

    int loop = 0;
    Detector denet;
    denet.LoadModeParamters("/home/data/code/catkin_ws/src/pillar_detect/model.pt");
    while (ros::ok())
    {
        KittiDataReader kittiDataReader("/home/data/dataset/KITTIDetection/data_object_velodyne/training/velodyne/",
                                        "/home/data/dataset/KITTIDetection/training/label_2/",
                                        "/home/data/dataset/KITTIDetection/calib/");
        int i = 0;
        loop++;
        while (ros::ok())
        {
            auto &data = kittiDataReader.getOnceData();
            if (!data.dataIsOk)
                break;

            std::cout << "loop:" << loop << " frame: " << i << "  " << data.objects.size() << " ";

            denet.Train(data.cloud, data.objects);

            if (i++ % 200 == 0)
            {
                denet.SaveModeParamters("/home/data/code/catkin_ws/src/pillar_detect/model.pt");
            }
        }
        denet.SaveModeParamters("/home/data/code/catkin_ws/src/pillar_detect/model.pt");
    }

    return 0;
}