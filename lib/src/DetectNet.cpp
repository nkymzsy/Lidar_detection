
#include <DetectNet.hpp>
#include <math.h>

class Gaussian2D
{
public:
    // 构造函数初始化均值和方差
    Gaussian2D(float r) :covariance_(Eigen::Matrix2f::Identity() * r / 3.0f)
    {
    }

    // 计算给定点的高斯分布值
    double operator()(Eigen::Vector2f diff) const
    {
        double exponent = -0.5 * diff.dot(covariance_.inverse() * diff);
        return std::exp(exponent);
    }

private:
    Eigen::Matrix2f covariance_; // 协方差矩阵
};

/// 从std::vector<Object> objs_gt生成heatmap真值
void Detector::BuildDetectionGroundTruth(const std::vector<Object> &objs,
                                        TensorMap &ground_truth)
{
    ground_truth = {
        {"heatmap", torch::zeros({Config::num_class, Config::bev_w, Config::bev_h})},
        {"center", torch::zeros({3, Config::bev_w, Config::bev_h})},
        {"dim", torch::zeros({3, Config::bev_w, Config::bev_h})},
        {"rot", torch::zeros({2, Config::bev_w, Config::bev_h})},
        {"mask", torch::zeros({Config::bev_w, Config::bev_h})}};

    auto &heat_map = ground_truth["heatmap"];
    auto &mean_map = ground_truth["center"];
    auto &dim_map = ground_truth["dim"];
    auto &rot_map = ground_truth["rot"];
    auto &mask_map = ground_truth["mask"];

    for (auto &obj : objs)
    {
        int label = 0;
        if (obj.label == 0 || obj.label == 1 || obj.label == 2)
            label = 0;
        else if (obj.label == 3 || obj.label == 4 || obj.label == 5)
            label = 1;
        else
            continue;

        auto center = PillarsBuilder::Point2Index(obj.position);

        if (center.x() < 0 || center.y() < 0 || center.x() >= Config::bev_w || center.y() >= Config::bev_h)
        {
            continue;
        }

        int range_x = ceil(obj.dimensions.x() / Config::pillar_x_size);
        int range_y = ceil(obj.dimensions.y() / Config::pillar_y_size);
        int range = std::max(range_x, range_y);
        float r = std::max(obj.dimensions.x(), obj.dimensions.y()) / 2;
        Gaussian2D gaussian(r);
        for (int i = center.x() - range; i < center.x() + range; i++)
        {
            for (int j = center.y() - range; j < center.y() + range; j++)
            {
                float detal_i = std::abs(i - center.x()) * Config::pillar_x_size;
                float detal_j = std::abs(j - center.y()) * Config::pillar_y_size;
                if (i < 0 || j < 0 || i >= Config::bev_w || j >= Config::bev_h)
                    continue;

                float p = gaussian(Eigen::Vector2f(detal_i, detal_j));
                if (p > heat_map.index({label, i, j}).item<float>())
                {
                    heat_map.index({label, i, j}) = p;
                }
            }
        }
        mean_map.index({torch::indexing::Slice(torch::indexing::None), center.x(), center.y()}) =
            torch::tensor({obj.position.x(), obj.position.y(), obj.position.z()});

        dim_map.index({torch::indexing::Slice(torch::indexing::None), center.x(), center.y()}) =
            torch::tensor({obj.dimensions.x(), obj.dimensions.y(), obj.dimensions.z()});

        rot_map.index({torch::indexing::Slice(torch::indexing::None), center.x(), center.y()}) =
            torch::tensor({sin(obj.heading), cos(obj.heading)});

        mask_map.index({center.x(), center.y()}) = 1.0f;
    }

    heat_map = heat_map.unsqueeze(0);
    mean_map = mean_map.unsqueeze(0);
    dim_map = dim_map.unsqueeze(0);
    rot_map = rot_map.unsqueeze(0);
    mask_map = mask_map.unsqueeze(0);
};

void Detector::BuildDetectionGroundTruth(const std::vector<DataPair> &data, TensorMap &ground_truth)
{

    for (int i = 0; i < data.size(); i++)
    {
        auto &obj = data[i].second;
        TensorMap groundtruthTemp;
        BuildDetectionGroundTruth(obj, groundtruthTemp);
        if (i == 0)
        {
            ground_truth = groundtruthTemp;
        }
        else
        {
            for (auto &[key, tensor] : ground_truth)
            {
                tensor = torch::cat({tensor, groundtruthTemp[key]}, 0);
            }
        }
    }
}

void Detector::Train(CloudType &cloud, const std::vector<Object> &objs_gt)
{
    TensorMap groundtruth;
    auto headmap = model->forward(cloud);
    BuildDetectionGroundTruth(objs_gt, groundtruth);
    auto loss = loss_function.forward(headmap, groundtruth);
    std::cout << "loss: " << loss.item<float>() << std::endl;
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
}

void Detector::Train(const std::vector<DataPair> &data, int accumulation_steps)
{
    static int i = 0;
    TensorMap groundtruth;
    BuildDetectionGroundTruth(data, groundtruth);

    auto headmap = model->forward(data);

    auto loss = loss_function.forward(headmap, groundtruth) / accumulation_steps;
    loss.backward();
    std::cout << "loss: " << loss.item<float>() << std::endl;

    if (++i % accumulation_steps == 0)
    {
        optimizer.step();
        optimizer.zero_grad();
    }
}

// 保存模型参数
void Detector::SaveModeParamters(const std::string &path) { torch::save(model, path); }

// 导入模型 参数
void Detector::LoadModeParamters(const std::string &path) { torch::load(model, path); }

Eigen::Vector3f Transform3DTensor2Eigen(const torch::Tensor &t)
{
    return Eigen::Vector3f(t[0].item<float>(), t[1].item<float>(), t[2].item<float>());
}

/**
 * @brief 将检测结果追加到对象列表中。
 *
 * @param output 包含检测结果的字典，键为 "center", "dim", "rot"，值为对应的张量。
 * @param indexs 索引张量，用于从输出张量中选择特定元素。
 * @param label 对象标签。
 * @param objs 存储对象的容器，用于追加新对象。
 */
void AppendObject(std::unordered_map<std::string, torch::Tensor> &output, const torch::Tensor &indexs, int label,
                  std::vector<Object> &objs)
{
    auto &means = output["center"];
    auto &dims = output["dim"];
    auto &rots = output["rot"];

    for (int i = 0; i < indexs.size(0); i++)
    {
        int x = indexs[i][0].item<int>();
        int y = indexs[i][1].item<int>();
        float heading = atan2(rots.index({0, x, y, 0}).item<float>(), rots.index({0, x, y, 1}).item<float>());
        objs.emplace_back(label, heading, Transform3DTensor2Eigen(means.index({0, x, y})),
                          Transform3DTensor2Eigen(dims.index({0, x, y})));
    }
}

void Detector::Infer(CloudType &cloud, std::vector<Object> &objs, float theshold)
{
    // 1. 得到网络输出结果
    auto output = model->forward(cloud);

    // 2. 做非极大值抑制
    auto heatmap = output["heatmap"];
    auto max_pooled =
        torch::nn::functional::max_pool2d(heatmap, torch::nn::functional::MaxPool2dFuncOptions(3).stride(1).padding(1));
    heatmap = (heatmap >= max_pooled).toType(torch::kFloat32) * heatmap;

    // 3. 提取出满足条件的索引
    auto car_indexs = torch::nonzero(heatmap[0][0].gt(theshold)).to(torch::kCPU);
    auto people_indexs = torch::nonzero(heatmap[0][1].gt(theshold)).to(torch::kCPU);

    // 3. 将output中的其他量转移到cpu并从B*C*H*W维度转换为B*H*W*C
    output["center"] = output["center"].permute({0, 2, 3, 1}).to(torch::kCPU);
    output["dim"] = output["dim"].permute({0, 2, 3, 1}).to(torch::kCPU);
    output["rot"] = output["rot"].permute({0, 2, 3, 1}).to(torch::kCPU);

    // 4. 将检测结果追加到对象列表中
    objs.clear();
    AppendObject(output, car_indexs, 0, objs);
    AppendObject(output, people_indexs, 1, objs);
}