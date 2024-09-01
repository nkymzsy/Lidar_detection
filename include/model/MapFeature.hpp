#pragma once

#include "Config.hpp"
#include "ResNet.hpp"
#include <Eigen/Core>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

class MapFeatureImpl: public torch::nn::Module
{
private:
    torch::Tensor bev_map;
    int pillar_final_feature_dim_;
    ResBEVBackboneConcat resnet;

public:
    MapFeatureImpl() : bev_map(torch::zeros({Config::bev_w + 1, Config::bev_h + 1, Config::pillar_feature_dim_out})) {
        register_module("resnet", resnet);
    }
    torch::Tensor forward(torch::Tensor pillars, std::vector<Eigen::Vector2i> &index)
    {
        // pillar 重新排序生成bevmap
        bev_map = torch::zeros({Config::bev_w, Config::bev_h, Config::pillar_feature_dim_out});
        // index转tensor张量
        torch::Tensor index_tensor = torch::from_blob(index.data(), {(int)index.size(), 2}, torch::kInt32);
        auto indices_y = index_tensor.select(1, 1).to(torch::kLong); // 第二列为 y 坐标
        auto indices_x = index_tensor.select(1, 0).to(torch::kLong); // 第一列为 x 坐标

        bev_map.index_put_({indices_x, indices_y, torch::indexing::Slice()}, pillars);
        // bev_map增加一维通道
        bev_map = bev_map.unsqueeze(0).permute({0, 3, 1, 2});;

        // 从BevMap 提取特征生成特征图
        auto feature_map = resnet->forward(bev_map);

        return feature_map;
    }
};

TORCH_MODULE(MapFeature);

