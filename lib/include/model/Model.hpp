#pragma once

#include <torch/torch.h>

#include "Datatype.hpp"
#include "model/CenterHead.hpp"
#include "model/Config.hpp"
#include "model/MapFeature.hpp"
#include "model/PillarsBuilder.hpp"
#include "model/PointNet.hpp"

class DetectNetImpl : public torch::nn::Module
{
private:
    using CloudType = pcl::PointCloud<PillarsBuilder::PointType>;
    PillarsBuilder pf_;
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
        auto [pillars, pillars_index] = pf_.BuildPillarsByCuda(cloud);
        auto pillarFeatures = pn_->forward(pillars);
        auto feature_map = mf_->forward(pillarFeatures, pillars_index);
        auto &head_output = ch_->forward(feature_map);
        return head_output;
    }

    std::unordered_map<std::string, at::Tensor> &
    forward(const std::vector<std::pair<pcl::PointCloud<PillarsBuilder::PointType>, std::vector<Object>>> &data)
    {
        auto [pillars, pillars_index] = pf_.BuildForClouds(data); // 训练的时候不用cuda加速了 尽量避免显存溢出 留给batchsize用
        auto pillarFeatures = pn_->forward(pillars);
        auto feature_map = mf_->forward(pillarFeatures, pillars_index, data.size());
        auto &head_output = ch_->forward(feature_map);
        return head_output;
    }
};

TORCH_MODULE(DetectNet);