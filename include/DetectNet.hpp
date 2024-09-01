
#pragma once

#include <torch/torch.h>

#include "Datatype.hpp"
#include "model/CenterHead.hpp"
#include "model/Config.hpp"
#include "model/MapFeature.hpp"
#include "model/PillarFeatureGenerate.hpp"
#include "model/PointNet.hpp"

class DetectNetImpl : public torch::nn::Module
{
private:
    using CloudType = pcl::PointCloud<PillarFeatureGenerate::PointType>;
    PillarFeatureGenerate pf_;
    PointNet pn_;
    MapFeature mf_;
    CenterHead ch_;

public:
    DetectNetImpl() {
        register_module("pn", pn_);
        register_module("mf", mf_);
        register_module("ch", ch_);
    }

    /// @brief 训练网络
    std::unordered_map<std::string, at::Tensor>& forward(CloudType &cloud);

};

TORCH_MODULE(DetectNet);

class Detector
{
private:
    using CloudType = pcl::PointCloud<PillarFeatureGenerate::PointType>;
    DetectNet model;
    std::vector<Object> objs_infer;

    torch::autograd::variable_list all_parameters;
    std::unique_ptr<torch::optim::Adam> optimizer;

public:
    Detector()
    {
        model->train();
        optimizer = std::make_unique<torch::optim::Adam>(model->parameters(), torch::optim::AdamOptions(0.0001));
    }

    void train(CloudType &cloud, std::vector<Object> objs_gt);
};