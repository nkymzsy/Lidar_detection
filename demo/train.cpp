#include "lib/include/DetectNet.hpp"
#include "tools/KittiReader.hpp"

int main(int argc, char **argv)
{
    int batch_size = 8;
    int epoches = 0;
    Detector denet;
    std::string model_path = "/home/data/code/catkin_ws/temp/60epoches_model.pt";
    denet.LoadModeParamters(model_path);
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
            auto data = kittiDataReader.getBatchData(batch_size);
            if (data && data->size() == batch_size)
            {
                std::cout << "epoches: " << epoches << "  curr: " << i++ << "  ";
                denet.Train(*data, 8);
                if (i % 20 == 0)
                {
                    denet.SaveModeParamters(model_path);
                }
            }
            else
            {
                break;
            }
        }
        denet.SaveModeParamters(model_path);
    }

    return 0;
}