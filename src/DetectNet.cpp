
#include <../include/DetectNet.hpp>
#include <math.h>

class Gaussian2D
{
public:
    // 构造函数初始化均值和方差
    Gaussian2D(Eigen::Vector2f mean, float range) : mean_(mean), covariance_(Eigen::Matrix2f::Identity() * range / 3.0f)
    {
    }

    // 计算给定点的高斯分布值
    double operator()(Eigen::Vector2f diff) const
    {
        double exponent = -0.5 * diff.dot(covariance_.inverse() * diff);
        double norm_factor = 1.0 / (2 * M_PI * std::sqrt(covariance_.determinant()));
        return norm_factor * std::exp(exponent);
    }

private:
    Eigen::Vector2f mean_;       // 均值向量
    Eigen::Matrix2f covariance_; // 协方差矩阵
};

/// 从std::vector<Object> objs_gt生成heatmap真值
void Detector::GenerateHeatMapGroundturth(std::vector<Object> objs_gt)
{
    ground_truth.heat_map = torch::zeros({Config::num_class, Config::bev_w, Config::bev_h});
    ground_truth.mean_map = torch::zeros({3, Config::bev_w, Config::bev_h});
    ground_truth.dim_map = torch::zeros({3, Config::bev_w, Config::bev_h});
    ground_truth.rot_map = torch::zeros({2, Config::bev_w, Config::bev_h});
    ground_truth.mask_map = torch::zeros({Config::bev_w, Config::bev_h});

    for (auto &obj : objs_gt)
    {
        int lable = 0;
        if (obj.lable == 0 || obj.lable == 1 || obj.lable == 2)
            lable = 0;
        else if (obj.lable == 3 || obj.lable == 4 || obj.lable == 5)
            lable = 1;
        else
            continue;

        auto index = PillarFeatureGenerate::Point2Index(obj.position);
        float range = std::sqrt(obj.dimensions.x() * obj.dimensions.x() + obj.dimensions.y() * obj.dimensions.y()) / 2;
        int range_x = (range + Config::pillar_x_size) / Config::pillar_x_size;
        int range_y = (range + Config::pillar_y_size) / Config::pillar_y_size;
        Gaussian2D gaussian(obj.position.block<2, 1>(0, 0), range);
        for (int i = index.x() - range_x; i < index.x() + range_x; i++)
        {
            for (int j = index.y() - range_y; j < index.y() + range_y; j++)
            {
                float detal_i = std::abs(i - index.x()) * Config::pillar_x_size;
                float detal_j = std::abs(j - index.y()) * Config::pillar_y_size;
                if (i < 0 || j < 0 || i >= Config::bev_w || j >= Config::bev_h)
                    continue;
                ground_truth.heat_map[lable][i][j] += gaussian(Eigen::Vector2f(detal_i, detal_j));

                ground_truth.mean_map[0][i][j] = obj.position.x();
                ground_truth.mean_map[1][i][j] = obj.position.y();
                ground_truth.mean_map[2][i][j] = obj.position.z();

                ground_truth.dim_map[0][i][j] = obj.dimensions.x();
                ground_truth.dim_map[1][i][j] = obj.dimensions.y();
                ground_truth.dim_map[2][i][j] = obj.dimensions.z();

                ground_truth.rot_map[0][i][j] = sin(obj.heading);
                ground_truth.rot_map[1][i][j] = cos(obj.heading);
                if (fabs(i - index.x()) < 3 && fabs(j - index.y()) < 3)
                    ground_truth.mask_map[i][j] = 1;
            }
        }
    }

    ground_truth.heat_map.to(device);
    ground_truth.mean_map.to(device);
    ground_truth.dim_map.to(device);
    ground_truth.rot_map.to(device);
    ground_truth.mask_map.to(device);
}

torch::Tensor Detector::HeatmapLoss(torch::Tensor heatmap)
{
    auto layer_output = heatmap.flatten().to(device);
    auto layer_target = ground_truth.heat_map.flatten().to(device);
    auto layer_loss = (layer_output - layer_target).abs().sum();
    return layer_loss;
}

torch::Tensor Detector::BoxLoss(torch::Tensor mean_map, torch::Tensor dim_map, torch::Tensor rot_map)
{
    torch::Tensor mask_3 = ground_truth.mask_map.flatten().tile(3).to(device);
    torch::Tensor mask_2 = ground_truth.mask_map.flatten().tile(2).to(device);
    auto mean_loss =
        torch::square((mean_map.flatten().to(device) - ground_truth.mean_map.flatten().to(device)) * mask_3).sum();
    auto dim_loss =
        torch::square(((dim_map.flatten().to(device) - ground_truth.dim_map.flatten().to(device)) * mask_3)).sum();
    auto rot_loss =
        torch::square((rot_map.flatten().to(device) - ground_truth.rot_map.flatten().to(device)) * mask_2).sum();
    return mean_loss + dim_loss + rot_loss;
}

void Detector::train(CloudType &cloud, std::vector<Object> objs_gt)
{
    auto headmap = model->forward(cloud);


    GenerateHeatMapGroundturth(objs_gt);
    auto loss = HeatmapLoss(headmap["heatmap"][0]);
    auto loss_box = BoxLoss(headmap["center"], headmap["dim"], headmap["rot"]);
    auto loss_total = 10 * loss + loss_box;

    std::cout << "loss: " << loss.item<float>() << " loss_box: " << loss_box.item<float>() << std::endl;

    optimizer->zero_grad();
    loss_total.backward();
    optimizer->step();
}

// 保存模型参数
void Detector::save(const std::string &path)
{
    torch::save(model, path);
}

// 导入模型 参数
void Detector::load(const std::string &path)
{
    torch::load(model, path);
}