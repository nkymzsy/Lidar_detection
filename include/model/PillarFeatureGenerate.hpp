// 基于哈希表产生pillar特征

#pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <torch/torch.h>
#include <unordered_map>

#include "Config.hpp"

class PillarFeatureGenerate
{
public:
    using PointType = pcl::PointXYZI;

private:
    struct Features
    {
        float data[Config::pillar_feature_dim];
    };

    std::vector<Features> pillar_features_;
    struct Vector2iHash
    {
        size_t operator()(const Eigen::Vector2i &index) const { return index(0) ^ index(1); }
    };
    std::unordered_map<Eigen::Vector2i, pcl::PointCloud<PointType>, Vector2iHash> pillars_;



    torch::Tensor pillar_features_tensor_;
    std::vector<Eigen::Vector2i> pillar_indices_;

public:
    PillarFeatureGenerate() {};
    ~PillarFeatureGenerate() {};

    static bool IsInRoi(const PointType &point)
    {
        return point.x > Config::roi_x_min && point.x < Config::roi_x_max && point.y > Config::roi_y_min &&
               point.y < Config::roi_y_max;
    }

    static Eigen::Vector2i Point2Index(const Eigen::Vector3f &point)
    {
        return Eigen::Vector2i((point.x() - Config::roi_x_min) / Config::pillar_x_size,
                               (point.y() - Config::roi_y_min) / Config::pillar_y_size);
    };


    void Generate(pcl::PointCloud<PointType> cloud)
    {
        pillars_.clear();
        for (const auto &point : cloud)
        {
            if (IsInRoi(point))
            {
                Eigen::Vector2i index = Point2Index(point.getVector3fMap());
                auto iter = pillars_.find(index);
                if (iter == pillars_.end())
                {
                    pillars_.insert(std::make_pair(index, pcl::PointCloud<PointType>()));
                    pillars_[index].reserve(Config::max_nums_in_pillar);
                }
                if (pillars_[index].size() < Config::max_nums_in_pillar)
                {
                    pillars_[index].push_back(point);
                }
            }
        }

        pillar_indices_.clear();
        pillar_indices_.reserve(pillars_.size());
        pillar_features_.clear();
        pillar_features_.reserve(Config::max_nums_in_pillar * pillars_.size());
        for (auto &pillar : pillars_)
        {
            auto &points = pillar.second;
            auto &index = pillar.first;
            auto &x = index(0);
            auto &y = index(1);
            Eigen::Vector3f mean(0, 0, 0);
            for (auto &point : points)
            {
                mean += point.getVector3fMap();
            }
            mean /= points.size();
            while (points.size() < Config::max_nums_in_pillar)
            {
                points.push_back(points[rand() % points.size()]);
            }

            pillar_indices_.emplace_back(index);
            for (auto &point : points)
            {
                Eigen::Vector3f pc = point.getVector3fMap() - mean;
                pillar_features_.emplace_back(Features{(float)index.x(), (float)index.y(), point.x, point.y, point.z,
                                                       point.intensity, pc.x(), pc.y(), pc.z()});
            }
        }
        // n * 32 * 9 的torch
        int n = pillar_features_.size() / Config::max_nums_in_pillar;
        pillar_features_tensor_ =
            torch::from_blob(reinterpret_cast<float *>(pillar_features_.data()),
                             {n, Config::max_nums_in_pillar, Config::pillar_feature_dim}, torch::kFloat32);
    };

    torch::Tensor &getPillars() { return pillar_features_tensor_; };

    std::vector<Eigen::Vector2i> &getPillarIndices() { return pillar_indices_; };
};
