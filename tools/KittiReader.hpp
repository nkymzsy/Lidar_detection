#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <utility>

#include "lib/include/Datatype.hpp"

// 定义点云类型
using PointXYZI = pcl::PointXYZI;

class KittiDataReader
{
    using DataPair = std::pair<pcl::PointCloud<PointXYZI>, std::vector<Object>>; 
public:

    std::map<std::string, int> label2Number = {
        {"Car", 0},     {"Van", 1},  {"Truck", 2}, {"Pedestrian", 3}, {"Person_sitting", 4},
        {"Cyclist", 5}, {"Tram", 6}, {"Misc", 7},  {"DontCare", 8}};

    std::shared_ptr<std::vector<DataPair>> getBatchData(int n)
    {
        auto data = std::make_shared<std::vector<DataPair>>();
        data->reserve(n);
        while (n--)
        {
            auto OnceData = getOnceData();
            if (OnceData)
            {
                data->emplace_back(std::move(*OnceData));
            }
            else
            {
                break;
            }
        }
        return data;
    }

    std::shared_ptr<DataPair> getOnceData()
    {
        std::shared_ptr<DataPair> cloudData(new DataPair);
        std::stringstream ss;
        ss << std::setw(6) << std::setfill('0') << currentIndex_++; // 生成六位数字，不足部分用0填充
        std::string formattedIndex = ss.str();
        if (readCalibration(calibPath_ + formattedIndex + ".txt") &&
            readPointCloud(pointCloudPath_ + formattedIndex + ".bin", *cloudData) &&
            readLabels(labelPath_ + formattedIndex + ".txt", *cloudData))
        {
            return cloudData;
        }
        return nullptr;
    }

    KittiDataReader(const std::string &pointCloudPath, const std::string &labelPath, const std::string &calibPath)
        : pointCloudPath_(pointCloudPath), labelPath_(labelPath), calibPath_(calibPath)
    {
    }

private:
    int currentIndex_ = 0;
    std::string pointCloudPath_;
    std::string labelPath_;
    std::string calibPath_;
    Eigen::Matrix4f Tr_velo_to_cam_;

    bool readPointCloud(std::string path, DataPair &cloudData)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open())
        {
            return false;
        }
        auto &cloud = cloudData.first;
        cloud.clear();
        pcl::PointXYZI p;
        while (file.read(reinterpret_cast<char *>(&p.x), sizeof(float)) &&
               file.read(reinterpret_cast<char *>(&p.y), sizeof(float)) &&
               file.read(reinterpret_cast<char *>(&p.z), sizeof(float)) &&
               file.read(reinterpret_cast<char *>(&p.intensity), sizeof(float)))
        {
            cloud.push_back(p);
        }

        file.close();
        return true;
    }

    bool readLabels(std::string path, DataPair &cloudData)
    {
        std::ifstream file(path);
        auto &objects = cloudData.second;
        objects.clear();
        if (!file.is_open())
        {
            return false;
        }

        std::string line;
        Eigen::Matrix4f Tr_cam_to_velo = Tr_velo_to_cam_.inverse();
        while (std::getline(file, line))
        {
            std::istringstream iss(line);
            std::string type;
            float truncated, occluded, alpha;
            float x1, y1, x2, y2;
            float h, w, l, x, y, z;
            float ry;

            if (!(iss >> type >> truncated >> occluded >> alpha >> x1 >> y1 >> x2 >> y2 >> h >> w >> l >> x >> y >> z >>
                  ry))
            {
                continue;
            }
            Eigen::Vector3f center =
                Tr_cam_to_velo.block<3, 3>(0, 0) * Eigen::Vector3f(x, y, z) + Tr_cam_to_velo.block<3, 1>(0, 3);
            objects.emplace_back(label2Number[type], ry + M_PI / 2, center, Eigen::Vector3f(l, w, h));
        }

        file.close();
        return true;
    }

    bool readCalibration(std::string path)
    {
        std::ifstream file(path);

        if (!file.is_open())
        {
            return false;
        }

        std::string line;
        while (std::getline(file, line))
        {
            std::istringstream iss(line);
            std::string key;
            iss >> key; // 获取键

            if (key == "Tr_velo_to_cam:")
            {
                float values[12];
                for (int i = 0; i < 12; ++i)
                {
                    iss >> values[i]; // 读取矩阵值
                }

                Tr_velo_to_cam_ << values[0], values[1], values[2], values[3], values[4], values[5], values[6],
                    values[7], values[8], values[9], values[10], values[11], 0, 0, 0, 1;
                break;
            }
        }

        file.close();
        return true;
    }
};
