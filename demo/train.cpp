#include "lib/include/DetectNet.hpp"
#include "tools/KittiReader.hpp"

int main(int argc, char **argv)
{
    int epoches = 0;
    Detector denet;
    denet.LoadModeParamters("/home/data/code/catkin_ws/src/pillar_detect/model.pt");
    while (1)
    {
        std::string cloud_path = "/home/data/dataset/KITTIDetection/data_object_velodyne/training/velodyne/";
        std::string label_path = "/home/data/dataset/KITTIDetection/training/label_2/";
        std::string calib_path = "/home/data/dataset/KITTIDetection/calib/";
        KittiDataReader kittiDataReader(cloud_path, label_path, calib_path);
        int i = 0;
        epoches++;
        while (1)
        {
            std::cout << "epoches: " << epoches << "  curr: " << i++ << "  ";
            auto data = kittiDataReader.getBatchData(8);
            if (data && data->size() == 8)
            {
                denet.Train(*data);
                if (i % 20 == 0)
                {
                    denet.SaveModeParamters("/home/data/code/catkin_ws/src/pillar_detect/model.pt");
                }
            }
        }
        denet.SaveModeParamters("/home/data/code/catkin_ws/src/pillar_detect/model.pt");
    }

    return 0;
}