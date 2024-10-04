#pragma once
#include "Datatype.hpp"
#include <torch/torch.h>

class Loss : public torch::nn::Module
{
public:
    Loss() : smooth_l1_loss(torch::nn::SmoothL1LossOptions().reduction(torch::kNone).beta(1 / 2.0f)) {}
    torch::Tensor forward(std::unordered_map<std::string, at::Tensor> pred,
                          std::unordered_map<std::string, at::Tensor> groundtruth)
    {
        torch::Tensor &cls_pred = pred["heatmap"];
        torch::Tensor &cls_real = groundtruth["heatmap"];

        torch::Tensor &bbox_mean_pred = pred["center"];
        torch::Tensor &bbox_mean_real = groundtruth["center"];

        torch::Tensor &bbox_dim_pred = pred["dim"];
        torch::Tensor &bbox_dim_real = groundtruth["dim"];

        torch::Tensor &bbox_heading_pred = pred["rot"];
        torch::Tensor &bbox_heading_real = groundtruth["rot"];

        torch::Tensor &mask = groundtruth["validmask"];
        int n_valid = mask.sum().item().toInt();

        cls_pred = cls_pred.to(device);
        cls_real = cls_real.to(device);
        bbox_mean_pred = bbox_mean_pred.to(device);
        bbox_mean_real = bbox_mean_real.to(device);
        bbox_dim_pred = bbox_dim_pred.to(device);
        bbox_dim_real = bbox_dim_real.to(device);
        bbox_heading_pred = bbox_heading_pred.to(device);
        bbox_heading_real = bbox_heading_real.to(device);
        mask = mask.to(device);

        // 1. bbox cls loss 修改版本的focal loss 详见论文
        const float eps = 1e-4;
        cls_pred = torch::clamp(torch::sigmoid(cls_pred), eps, 1.0 - eps);
        auto cls_loss =
            torch::where(cls_real > 0.99, torch::pow(1 - cls_pred, alpha) * torch::log(cls_pred),
                         torch::pow(1 - cls_real, beta) * torch::pow(cls_pred, alpha) * torch::log(1 - cls_pred));
        cls_loss = cls_loss.sum();

        // 2. regression mean loss
        auto mask3d = mask.unsqueeze(1).expand_as(bbox_mean_pred);
        auto mean_loss = smooth_l1_loss(bbox_mean_pred, bbox_mean_real) * mask3d;
        mean_loss = mean_loss.sum();

        // 2. regression dim loss
        auto dim_loss = smooth_l1_loss(bbox_dim_pred, bbox_dim_real) * mask3d;
        dim_loss = dim_loss.sum();

        // 3. regression rot loss
        auto mask2d = mask.unsqueeze(1).expand_as(bbox_heading_pred);
        auto rot_loss = smooth_l1_loss(bbox_heading_pred, bbox_heading_real) * mask2d;
        rot_loss = rot_loss.sum();

        // 4. total loss
        if (n_valid > 0)
            return (-cls_w * cls_loss + mean_w * mean_loss + dim_w * dim_loss + rot_w * rot_loss) / n_valid;

        return -cls_w * cls_loss;
    }

private:
    float alpha = 3;
    float beta = 3;
    float cls_w = 2, mean_w = 1, dim_w = 1, rot_w = 2;
    torch::nn::SmoothL1Loss smooth_l1_loss;
    torch::Device device = torch::Device(torch::kCUDA);
};