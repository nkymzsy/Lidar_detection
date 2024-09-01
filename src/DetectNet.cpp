
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
torch::Tensor GenerateHeatMapGroundturth(std::vector<Object> objs_gt)
{
    torch::Tensor heat_map = torch::zeros({Config::num_class,  Config::bev_w, Config::bev_h});
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
        int range_x = range / Config::pillar_x_size;
        int range_y = range / Config::pillar_y_size;
        Gaussian2D gaussian(obj.position.block<2, 1>(0, 0), range);
        for (int i = index.x() - range_x; i < index.x() + range_x; i++)
        {
            for (int j = index.y() - range_y; j < index.y() + range_y; j++)
            {
                float detal_i = std::abs(i - index.x()) * Config::pillar_x_size;
                float detal_j = std::abs(j - index.y()) * Config::pillar_y_size;
                if(i < 0 || j < 0 || i >= Config::bev_w || j >= Config::bev_h) 
                    continue;
                heat_map[lable][i][j] += gaussian(Eigen::Vector2f(detal_i, detal_j));
            }
        }
    }
    return heat_map;
}

torch::Tensor HeatmapLoss(torch::Tensor heatmap_gt, torch::Tensor heatmap)
{
    auto loss_fn = torch::nn::CrossEntropyLoss();
    auto layer_output = heatmap.flatten();
    auto layer_target = heatmap_gt.flatten();
    auto layer_loss = (layer_output - layer_target).abs().sum();

    return layer_loss;
}

std::unordered_map<std::string, at::Tensor>& DetectNetImpl::forward(CloudType &cloud)
{
    pf_.Generate(cloud);
    auto pillars = pf_.getPillars();
    auto &pillars_index = pf_.getPillarIndices();
    auto pillarFeatures = pn_->forward(pillars);
    auto feature_map =  mf_->forward(pillarFeatures, pillars_index);
    auto & headmap = ch_->forward(feature_map);

    return headmap;
}

void Detector::train(CloudType &cloud, std::vector<Object> objs_gt)
{

    auto headmap = model->forward(cloud);
    auto heatmap = headmap["heatmap"];
    auto heatmap_gt = GenerateHeatMapGroundturth(objs_gt);
    auto loss = HeatmapLoss(heatmap_gt, heatmap[0]);
    std::cout << "loss: " << loss << std::endl;

    optimizer->zero_grad();
    loss.backward();
    optimizer->step();
}
