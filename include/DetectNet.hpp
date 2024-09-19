#pragma once

#include "model/Model.hpp"
#include "model/Loss.hpp"

class Detector
{
private:
    using CloudType = pcl::PointCloud<PillarFeatureGenerate::PointType>;
    DetectNet model;
    Loss loss_function;
    
    std::vector<Object> objs_infer;
    std::unique_ptr<torch::optim::Adam> optimizer;
    torch::Device device = torch::Device(torch::kCUDA);

    std::unordered_map<std::string, torch::Tensor> GenerateHeatMapGroundturth(std::vector<Object> objs_gt);

public:
    Detector()
    {
        optimizer = std::make_unique<torch::optim::Adam>(model->parameters(), torch::optim::AdamOptions(1e-7));
    }

    void train(CloudType &cloud, std::vector<Object> objs_gt);
    std::vector<Object> infer(CloudType &cloud);
    void save(const std::string &path);
    void load(const std::string &path);
};