
#pragma once
#include <torch/torch.h>

class SeparateHeadImpl : public torch::nn::Module
{
public:
    SeparateHeadImpl(int64_t input_channels, int64_t output_channels, int num_conv, float init_bias = -2.19f,
                     bool use_bias = false)
    {

        for (int64_t k = 0; k < num_conv - 1; ++k)
        {
            net->push_back(torch::nn::Conv2d(
                torch::nn::Conv2dOptions(input_channels, input_channels, 3).stride(1).padding(1).bias(use_bias)));
            register_module("Conv2d_" + std::to_string(k), net->modules().back());

            net->push_back(torch::nn::BatchNorm2d(input_channels));
            register_module("BatchNorm2d_" + std::to_string(k), net->modules().back());

            net->push_back(torch::nn::ReLU());
            register_module("ReLU_" + std::to_string(k), net->modules().back());
        }
        net->push_back(torch::nn::Conv2d(
            torch::nn::Conv2dOptions(input_channels, output_channels, 3).stride(1).padding(1).bias(true)));
        register_module("Conv2d_last", net->modules().back());
    };
    torch::Tensor forward(torch::Tensor x) { return net->forward(x); }

private:
    torch::nn::Sequential net;
};

TORCH_MODULE(SeparateHead);

class CenterHeadImpl : public torch::nn::Module
{
private:
    std::vector<std::string> head_names = {"heatmap", "center", "dim", "rot"};
    std::unordered_map<std::string, std::unordered_map<std::string, int>> sep_head_dict = {
        {"heatmap", {{"out_channels", 2}, {"num_conv", 1}}},
        {"center", {{"out_channels", 3}, {"num_conv", 1}}},
        {"dim", {{"out_channels", 3}, {"num_conv", 1}}},
        {"rot", {{"out_channels", 2}, {"num_conv", 1}}}};

    torch::nn::Sequential shared_conv;
    std::unique_ptr<SeparateHead> heads[4];
    std::unordered_map<std::string, torch::Tensor> head_outputs;

public:
    CenterHeadImpl(int64_t input_channels = 128, int num_class = 2, float init_bias = -2.19f, bool use_bias = false)
    {

        for (const auto &pair : sep_head_dict)
        {
            const auto &name = pair.first;
            const auto &config = pair.second;
        }

        shared_conv = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, 64, 3).stride(1).padding(1).bias(true)),
            torch::nn::BatchNorm2d(64), torch::nn::ReLU());
        register_module("shared_conv", shared_conv );

        for(int i = 0; i < head_names.size(); ++i)
        {
            heads[i] = std::make_unique<SeparateHead>(64, sep_head_dict[head_names[i]]["out_channels"], init_bias, use_bias);
            register_module("head" + std::to_string(i), *heads[i] );
        }
    }

    std::unordered_map<std::string, torch::Tensor> &forward(torch::Tensor x)
    {
        x = shared_conv->forward(x);
        for (int i = 0; i < 4; ++i)
        {
            head_outputs[head_names[i]] = (*heads[i])->forward(x);
        }
        return head_outputs;
    }
};


TORCH_MODULE(CenterHead);