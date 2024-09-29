#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include "Config.hpp"

class BottleNeckImpl : public torch::nn::Module
{
public:
    BottleNeckImpl(int64_t in_channels, int64_t out_channels, int64_t stride = 1)
        : Module(),
          residual_function(torch::nn::Sequential(
              torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1).bias(false)),
              torch::nn::BatchNorm2d(out_channels), torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
              torch::nn::Conv2d(
                  torch::nn::Conv2dOptions(out_channels, out_channels, 3).stride(stride).padding(1).bias(false)),
              torch::nn::BatchNorm2d(out_channels), torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
              torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, 1).bias(false)),
              torch::nn::BatchNorm2d(out_channels * 1)))
    {
        if (stride != 1 || in_channels != out_channels * 1)
        {
            shortcut = torch::nn::Sequential(
                torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(in_channels, out_channels * 1, 1).stride(stride).bias(false)),
                torch::nn::BatchNorm2d(out_channels * 1));
            register_module("shortcut", shortcut);
        }
        register_module("residual_function", residual_function);
 
    }
    torch::Tensor forward(torch::Tensor x)
    {
        auto residual = residual_function->forward(x);
        if (shortcut.is_empty())
            return torch::relu_(residual);
        auto out = residual + shortcut->forward(x);
        return torch::relu_(out);
    }

private:
    torch::nn::Sequential residual_function{nullptr};
    torch::nn::Sequential shortcut{nullptr};
};

TORCH_MODULE(BottleNeck);

class ResBEVBackboneConcatImpl : public torch::nn::Module
{
private:
    int input_channels;
    std::vector<int64_t> layer_nums = {2, 2, 3, 3, 2};
    std::vector<int64_t> layer_strides = {2, 2, 2, 2, 2};
    std::vector<int64_t> num_filters = {32, 48, 64, 96, 128};
    std::vector<int64_t> upsample_strides = {2, 4, 8, 16, 32};
    std::vector<torch::nn::Sequential> blocks;
    torch::nn::Sequential fusion;
    torch::nn::Sequential attention_w;
    int64_t num_bev_features;

    int64_t sum(const std::vector<int64_t> &vec)
    {
        int64_t total = 0;
        for (auto v : vec)
        {
            total += v;
        }
        return total;
    }

public:
    ResBEVBackboneConcatImpl(int _input_channels = Config::pillar_feature_dim_out) : input_channels(_input_channels)
    {
        int64_t num_levels = layer_nums.size();
        std::vector<int64_t> c_in_list = {input_channels};
        c_in_list.insert(c_in_list.end(), num_filters.begin(), num_filters.end() - 1);

        for (int64_t idx = 0; idx < num_levels; ++idx)
        {
            torch::nn::Sequential cur_layers(
                torch::nn::ZeroPad2d(1),
                torch::nn::Conv2d(torch::nn::Conv2dOptions(c_in_list[idx], num_filters[idx], 3)
                                      .stride(layer_strides[idx])
                                      .padding(0)
                                      .bias(false)),
                torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_filters[idx]).eps(1e-3).momentum(0.01)),
                torch::nn::ReLU());

            for (int64_t k = 0; k < layer_nums[idx]; ++k)
            {
                cur_layers->push_back(BottleNeck(num_filters[idx], num_filters[idx]));
            }

            blocks.push_back(torch::nn::Sequential(cur_layers));
            register_module("blocks" + std::to_string(idx), blocks.back());
        }

        fusion =
            torch::nn::Sequential(torch::nn::Conv2d(torch::nn::Conv2dOptions(sum(num_filters), 128, 1).bias(false)),
                                  torch::nn::BatchNorm2d(128), torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));

        attention_w = torch::nn::Sequential(torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 1).bias(false)),
                                            torch::nn::BatchNorm2d(128));

        register_module("fusion", fusion);
        register_module("attention_w", attention_w);
    }

    torch::Tensor forward(torch::Tensor spatial_features)
    {
        std::vector<torch::Tensor> ups;
        torch::Tensor x = spatial_features;

        for (int64_t i = 0; i < blocks.size(); ++i)
        {
            x = blocks[i]->forward(x);

            int64_t stride = static_cast<int64_t>(spatial_features.size(2) / x.size(2));
            ups.push_back(torch::upsample_bilinear2d(x, {spatial_features.size(2), spatial_features.size(3)}, false));
        }

        x = torch::cat(ups, /*dim=*/1);
        x = fusion->forward(x);
        torch::Tensor w_x = torch::softmax(attention_w->forward(x), /*dim=*/1);
        x = w_x * x;

        return x;
    };
};

TORCH_MODULE(ResBEVBackboneConcat);