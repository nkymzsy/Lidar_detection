#pragma once
#include <torch/torch.h>
namespace Config
{
    constexpr float pillar_x_size = 0.2;
    constexpr float pillar_y_size = 0.2;

    constexpr float roi_x_min = 0;
    constexpr float roi_x_max = 50;
    constexpr float roi_y_min = -40;
    constexpr float roi_y_max = 40;

    constexpr int max_nums_in_pillar = 16;
    constexpr int pillar_feature_dim = 9;       // pillar特征的输入维度
    constexpr int pillar_feature_dim_out = 64;  // pillar特征的输出维度

    constexpr int bev_w = (Config::roi_x_max - Config::roi_x_min + Config::pillar_x_size) / Config::pillar_x_size;
    constexpr int bev_h = (Config::roi_y_max - Config::roi_y_min + Config::pillar_y_size) / Config::pillar_y_size;

    constexpr int num_class = 2;

}