#pragma once

#include "model/Model.hpp"

class Detector
{
private:
    using CloudType = pcl::PointCloud<PillarFeatureGenerate::PointType>;
    DetectNet model;
    std::vector<Object> objs_infer;
    std::unique_ptr<torch::optim::Adam> optimizer;
    torch::Device device = torch::Device(torch::kCUDA);

    struct GroundTruth
    {
        torch::Tensor mask_map;
        torch::Tensor heat_map;
        torch::Tensor mean_map;
        torch::Tensor dim_map;
        torch::Tensor rot_map;
    } ground_truth;

    void GenerateHeatMapGroundturth(std::vector<Object> objs_gt);
    torch::Tensor HeatmapLoss(torch::Tensor heatmap);
    torch::Tensor BoxLoss(torch::Tensor mean_map, torch::Tensor dim_map, torch::Tensor rot_map);

public:
    Detector()
    {
        optimizer = std::make_unique<torch::optim::Adam>(model->parameters(), torch::optim::AdamOptions(0.0001));
    }

    void train(CloudType &cloud, std::vector<Object> objs_gt);
    void save(const std::string &path);
    void load(const std::string &path);
};