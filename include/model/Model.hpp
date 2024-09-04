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
    torch::Device device = torch::Device(torch::kCUDA);
public:
    DetectNetImpl()
    {
        pn_->to(device);
        mf_->to(device);
        ch_->to(device);

        register_module("pn", pn_);
        register_module("mf", mf_);
        register_module("ch", ch_);
    }

    std::unordered_map<std::string, at::Tensor> &forward(CloudType &cloud)
    {
        auto [pillars, pillars_index] = pf_.Generate(cloud);
        auto pillarFeatures = pn_->forward(pillars);
        auto feature_map = mf_->forward(pillarFeatures, pillars_index);
        auto &head_output = ch_->forward(feature_map);
        return head_output;
    }
};

TORCH_MODULE(DetectNet);