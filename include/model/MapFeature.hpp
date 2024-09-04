#pragma once

#include "Config.hpp"
#include "ResNet.hpp"
#include <Eigen/Core>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

class MapFeatureImpl : public torch::nn::Module
{
private:
    torch::Tensor bev_map;
    int pillar_final_feature_dim_;
    ResBEVBackboneConcat resnet;
    torch::Device device = torch::Device(torch::kCUDA);

public:
    MapFeatureImpl() : bev_map(torch::zeros({Config::bev_w + 1, Config::bev_h + 1, Config::pillar_feature_dim_out}))
    {
        register_module("resnet", resnet);
    }
    torch::Tensor forward(torch::Tensor pillars, torch::Tensor index)
    {
        // pillar 重新排序生成bevmap
        bev_map = torch::zeros({Config::bev_w, Config::bev_h, Config::pillar_feature_dim_out}).to(device);

        // scatter
        auto indices_y = index.select(1, 1).to(torch::kLong).to(device); // 第二列为 y 坐标
        auto indices_x = index.select(1, 0).to(torch::kLong).to(device); // 第一列为 x 坐标
        bev_map.index_put_({indices_x, indices_y, torch::indexing::Slice()}, pillars);

        // bev_map增加一维通道
        bev_map = bev_map.unsqueeze(0).permute({0, 3, 1, 2});

        // 从BevMap 提取特征生成特征图
        auto feature_map = resnet->forward(bev_map);

        return feature_map;
    }
};

TORCH_MODULE(MapFeature);

