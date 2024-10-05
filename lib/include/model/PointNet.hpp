// pointnet 网络 用于提取点云特征
#pragma once

#include <torch/torch.h>

#include "Config.hpp"


class PointNetImpl : public torch::nn::Module
{
public:
    PointNetImpl()
    {
        conv[0] = register_module("conv1", torch::nn::Conv1d(torch::nn::Conv1dOptions(9, 16, 1)));
        conv[1] = register_module("conv2", torch::nn::Conv1d(torch::nn::Conv1dOptions(16, 32, 1)));
        conv[2] = register_module("conv3", torch::nn::Conv1d(torch::nn::Conv1dOptions(32, Config::pillar_feature_dim_out, 1)));
        relu = register_module("relu", torch::nn::ReLU(torch::nn::ReLUOptions().inplace(false)));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = x.view({-1, 9});
        x = x.unsqueeze(-1);
        x = relu->forward(conv[0]->forward(x));
        x = relu->forward(conv[1]->forward(x));
        x = relu->forward(conv[2]->forward(x));
        x = x.view({-1, Config::max_nums_in_pillar, Config::pillar_feature_dim_out});
        torch::Tensor max_result = x.amax({1}, false);
        return max_result;
    }

private:
    torch::nn::Conv1d conv[3] = {nullptr, nullptr, nullptr};
    torch::nn::ReLU relu = nullptr;
};

TORCH_MODULE(PointNet);