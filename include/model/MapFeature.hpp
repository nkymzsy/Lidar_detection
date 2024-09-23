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
    torch::Tensor forward(torch::Tensor pillars, torch::Tensor index, int batch_dim = 1)
    {
        // pillar 重新排序生成bevmap
        bev_map = torch::zeros({batch_dim, Config::bev_w, Config::bev_h, Config::pillar_feature_dim_out}).to(device);

        // scatter
        auto indices_batch = index.select(1, 0).to(torch::kLong).to(device); // 第一列为 batch维度
        auto indices_x = index.select(1, 1).to(torch::kLong).to(device);     // 第一列为 x 坐标
        auto indices_y = index.select(1, 2).to(torch::kLong).to(device);     // 第二列为 y 坐标
        bev_map.index_put_({indices_batch, indices_x, indices_y, torch::indexing::Slice()}, pillars);

        // bev_map从BHWC张量转换为BCHW张量
        bev_map = bev_map.permute({0, 3, 1, 2});

        // 从BevMap 通过主干网络提取特征生成特征图
        auto feature_map = resnet->forward(bev_map);

        return feature_map;
    }
};

TORCH_MODULE(MapFeature);

