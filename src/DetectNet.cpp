
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
        return std::exp(exponent);
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

        auto center = PillarFeatureGenerate::Point2Index(obj.position);

        int range_x = (obj.dimensions.x() + Config::pillar_x_size) / Config::pillar_x_size;
        int range_y = (obj.dimensions.y() + Config::pillar_y_size) / Config::pillar_y_size;
        float range = std::max(obj.dimensions.x(), obj.dimensions.y()) / 2;
        Gaussian2D gaussian(obj.position.block<2, 1>(0, 0), range);
        for (int i = center.x() - range_x; i < center.x() + range_x; i++)
        {
            for (int j = center.y() - range_y; j < center.y() + range_y; j++)
            {
                float detal_i = std::abs(i - center.x()) * Config::pillar_x_size;
                float detal_j = std::abs(j - center.y()) * Config::pillar_y_size;
                if (i < 0 || j < 0 || i >= Config::bev_w || j >= Config::bev_h)
                    continue;

                float p = gaussian(Eigen::Vector2f(detal_i, detal_j));
                if (p > ground_truth.heat_map[lable][i][j].item<float>())
                {
                    ground_truth.heat_map[lable][i][j] = p;

                    ground_truth.mean_map[0][i][j] = obj.position.x();
                    ground_truth.mean_map[1][i][j] = obj.position.y();
                    ground_truth.mean_map[2][i][j] = obj.position.z();

                    ground_truth.dim_map[0][i][j] = obj.dimensions.x();
                    ground_truth.dim_map[1][i][j] = obj.dimensions.y();
                    ground_truth.dim_map[2][i][j] = obj.dimensions.z();

                    ground_truth.rot_map[0][i][j] = sin(obj.heading);
                    ground_truth.rot_map[1][i][j] = cos(obj.heading);
                    if (fabs(i - center.x()) < 2 && fabs(j - center.y()) < 2)
                        ground_truth.mask_map[i][j] = p;
                }
            }
        }
    }

    ground_truth.heat_map.to(device);
    ground_truth.mean_map.to(device);
    ground_truth.dim_map.to(device);
    ground_truth.rot_map.to(device);
    ground_truth.mask_map.to(device);
};

torch::Tensor Detector::HeatmapLoss(torch::Tensor heatmap)
{
    auto layer_output = heatmap.flatten().to(device);
    auto layer_target = ground_truth.heat_map.flatten().to(device);
    auto layer_loss = torch::square((layer_output - layer_target)).sum();
    return layer_loss;
}

torch::Tensor Detector::BoxLoss(torch::Tensor mean_map, torch::Tensor dim_map, torch::Tensor rot_map)
{
    torch::Tensor mask_3 = ground_truth.mask_map.flatten().tile(3).to(device);
    torch::Tensor mask_2 = ground_truth.mask_map.flatten().tile(2).to(device);
    auto mean_loss =
        torch::square(2 * (mean_map.flatten().to(device) - ground_truth.mean_map.flatten().to(device)) * mask_3).sum();
    auto dim_loss =
        torch::square(1.5 * (dim_map.flatten().to(device) - ground_truth.dim_map.flatten().to(device)) * mask_3).sum();
    auto rot_loss =
        torch::square(2 * (rot_map.flatten().to(device) - ground_truth.rot_map.flatten().to(device)) * mask_2).sum();
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
void Detector::save(const std::string &path) { torch::save(model, path); }

// 导入模型 参数
void Detector::load(const std::string &path) { torch::load(model, path); }

// 定义一个函数来获取满足条件的点的索引
std::vector<Eigen::Vector2i> findLocalMaxima(torch::Tensor tensor, double threshold = 0.6)
{
    std::vector<Eigen::Vector2i> result;
    int64_t n = tensor.size(0);
    int64_t m = tensor.size(1);

    // 遍历每个元素
    for (int64_t i = 1; i < n - 1; ++i)
    {
        for (int64_t j = 1; j < m - 1; ++j)
        {
            float value = tensor[i][j].item<float>();
            if (value <= threshold)
                continue;

            bool isLocalMax = true;
            // 检查3x3邻域内的所有点
            for (int64_t di = -1; di <= 1; ++di)
            {
                for (int64_t dj = -1; dj <= 1; ++dj)
                {
                    if (tensor[i + di][j + dj].item<float>() > value)
                    {
                        isLocalMax = false;
                        break;
                    }
                }
                if (!isLocalMax)
                    break;
            }

            // 如果当前点是局部极大值，则添加到结果列表
            if (isLocalMax)
            {
                result.emplace_back(i, j);
            }
        }
    }

    return result;
}

// 定义一个函数来从3*n*m的Tensor中筛选出指定索引的三维值
Eigen::Vector3f getValuesFromTensor(const at::Tensor &tensor, const Eigen::Vector2i &index)
{
    int64_t n = tensor.size(1); // 获取第二维度的大小
    int64_t m = tensor.size(2); // 获取第三维度的大小

    // 检查索引是否有效
    int64_t x = index(0);
    int64_t y = index(1);
    if (x < 0 || x >= n || y < 0 || y >= m)
    {
        throw std::out_of_range("Index out of bounds");
    }

    // 提取三维值
    at::Tensor values = tensor.select(1, x).select(1, y);

    // 将三个值存储在 Eigen::Vector3f 中
    Eigen::Vector3f result;
    result << values[0].item<float>(), values[1].item<float>(), values[2].item<float>();

    return result;
}

// 从tensor中解包角度
float getRotFromTensor(const at::Tensor &tensor, const Eigen::Vector2i &index)
{
    int64_t n = tensor.size(1); // 获取第二维度的大小
    int64_t m = tensor.size(2); // 获取第三维度的大小

    // 检查索引是否有效
    int64_t x = index(0);
    int64_t y = index(1);
    if (x < 0 || x >= n || y < 0 || y >= m)
    {
        throw std::out_of_range("Index out of bounds");
    }

    at::Tensor values = tensor.select(1, x).select(1, y);
    float sinx = values[0].item<float>();
    float cosx = values[1].item<float>();

    // 求解角度
    float result = atan2(sinx, cosx);
    return result;
}

// 从mean dim rot 中解包出object
std::vector<Object> UnpackObject(int lable, std::vector<Eigen::Vector2i> indexs, torch::Tensor means,
                                 torch::Tensor dims, torch::Tensor rots)
{
    std::vector<Object> objs;
    objs.reserve(indexs.size());
    for (auto index : indexs)
    {
        Eigen::Vector3f pos(getValuesFromTensor(means, index));
        Eigen::Vector3f dim(getValuesFromTensor(dims, index));
        float heading = getRotFromTensor(rots, index);
        objs.emplace_back(lable, heading, pos, dim);
    }
    return objs;
}

std::vector<Object> Detector::infer(CloudType &cloud)
{
    auto map = model->forward(cloud);
    auto heatmap = map["heatmap"][0];

    auto car = findLocalMaxima(heatmap[0], 0.8);
    auto poepole = findLocalMaxima(heatmap[1], 0.8);

    std::vector<Object> objs(UnpackObject(0, car, map["center"][0].to(torch::kCPU), map["dim"][0].to(torch::kCPU),
                                          map["rot"][0].to(torch::kCPU)));
    auto objs_p = UnpackObject(1, poepole, map["center"][0].to(torch::kCPU), map["dim"][0].to(torch::kCPU),
                               map["rot"][0].to(torch::kCPU));
    objs.insert(objs.end(), objs_p.begin(), objs_p.end());

    return objs;
}