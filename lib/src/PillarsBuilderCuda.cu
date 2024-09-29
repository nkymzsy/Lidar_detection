#include <model/PillarsBuilderCuda.h>

#include <Eigen/Core>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

__device__ bool IsInRoi(float x, float y)
{
    return x > Config::roi_x_min && x < Config::roi_x_max && y > Config::roi_y_min && y < Config::roi_y_max;
}

__device__ int Point2Index(float x, float y)
{
    return (x - Config::roi_x_min) / Config::pillar_x_size * Config::bev_h +
           (y - Config::roi_y_min) / Config::pillar_y_size;
};

__device__ Eigen::Vector2i Point2Index2d(float x, float y)
{
    return Eigen::Vector2i((x - Config::roi_x_min) / Config::pillar_x_size,
                           (y - Config::roi_y_min) / Config::pillar_y_size);
};

class PillarsBuilderCuda
{
public:
    struct PillarInfo
    {
        Eigen::Vector3f mean; // 记录pillar中心点坐标
        int num;              // 记录pillar中的有效点数目
        int idx;              // 记录pillar对应在稠密张量中的索引
    };

private:
    thrust::device_vector<Eigen::Matrix<float, 1, 8>> raw_cloud_; // 原始点云 按照PCL XYZI格式的设置为8个float

    const int pillars_nums_ = Config::bev_h * Config::bev_w; // pill的数目
    thrust::device_vector<PillarInfo> pillars_;

    thrust::device_vector<Eigen::Vector2i> pillar_idx_;
    thrust::device_vector<Eigen::Matrix<float, 1, 9>> pillar_feature_;

public:
    PillarsBuilderCuda();
    void BuildPillarsFeature(const float *data, int point_num);
    auto &GetPillarIndex() { return pillar_idx_; }
    auto &GetPillarFeatureData() { return pillar_feature_; }
};

PillarsBuilderCuda::PillarsBuilderCuda()
{
    pillars_.resize(pillars_nums_);
};

void PillarsBuilderCuda::BuildPillarsFeature(const float *data, int point_num)
{
    // 1. 数据传输到 GPU
    raw_cloud_.resize(point_num);
    auto data_reint = reinterpret_cast<const Eigen::Matrix<float, 1, 8> *>(data);
    thrust::copy(data_reint, data_reint + point_num, raw_cloud_.begin());

    // 2. 重置所有Pillar的数据
    thrust::for_each_n(pillars_.begin(), pillars_nums_,
                       [] __device__(PillarInfo & pillar)
                       {
                           pillar.mean = Eigen::Vector3f::Zero();
                           pillar.num = 0;
                       });

    // 3. 计算每个Pillar的sum和num

    auto pillars_ptr = thrust::raw_pointer_cast(pillars_.data());
    thrust::for_each_n(raw_cloud_.begin(), point_num,
                       [=] __device__(auto &point)
                       {
                           if (IsInRoi(point[0], point[1]))
                           {
                               int index = Point2Index(point[0], point[1]);
                               atomicAdd(&pillars_ptr[index].num, 1);
                               atomicAdd(&pillars_ptr[index].mean.x(), point[0]);
                               atomicAdd(&pillars_ptr[index].mean.y(), point[1]);
                               atomicAdd(&pillars_ptr[index].mean.z(), point[2]);
                           }
                       });

    // 4. 计算每个Pillar的mean并得到每个有效pillar在稠密张量中对应的索引
    thrust::device_vector<int> valid_pillars(1, 0);
    auto valid_pillars_ptr = thrust::raw_pointer_cast(valid_pillars.data());
    thrust::for_each_n(pillars_.begin(), pillars_nums_,
                       [=] __device__(PillarInfo & pillar)
                       {
                           if (pillar.num > 0)
                           {
                               pillar.mean = pillar.mean / pillar.num;
                               pillar.idx = atomicAdd(valid_pillars_ptr, 1);
                               pillar.num = 0; // 这里是为了下面重新用这个量记录有效的点数量
                           }
                           else
                           {
                               pillar.idx = -1;
                           }
                       });

    // 5. 构建稠密pillar特征
    int pillar_num = valid_pillars[0];
    pillar_feature_.resize(pillar_num * Config::max_nums_in_pillar, Eigen::Matrix<float, 1, 9>::Zero());
    pillar_idx_.resize(pillar_num, Eigen::Vector2i::Zero());
    auto pillar_feature_ptr = thrust::raw_pointer_cast(pillar_feature_.data());
    auto pillar_idx_ptr = thrust::raw_pointer_cast(pillar_idx_.data());
    thrust::for_each_n(raw_cloud_.begin(), point_num,
                       [=] __device__(auto &point)
                       {
                           if (IsInRoi(point[0], point[1]))
                           {
                               int index = Point2Index(point[0], point[1]);
                               int idx_curr = atomicAdd(&pillars_ptr[index].num, 1);
                               if (idx_curr < Config::max_nums_in_pillar)
                               {
                                   int idx_feature = pillars_ptr[index].idx * Config::max_nums_in_pillar + idx_curr;
                                   auto index2d = Point2Index2d(point[0], point[1]);

                                   pillar_feature_ptr[idx_feature] << index2d.x(), index2d.y(), point[0], point[1],
                                       point[2], point[4], point[0] - pillars_ptr[index].mean.x(),
                                       point[1] - pillars_ptr[index].mean.y(), point[2] - pillars_ptr[index].mean.z();
                                   if (idx_curr == 0)
                                   {
                                       pillar_idx_ptr[pillars_ptr[index].idx] << index2d.x(), index2d.y();
                                   }
                               }
                           }
                       });
}

std::tuple<float *, int *, int> BuildPillarsFeature(const float *data, int point_num)
{
    static PillarsBuilderCuda pillars_builder_cuda;
    pillars_builder_cuda.BuildPillarsFeature(data, point_num);
    return std::make_tuple(
        reinterpret_cast<float *>(thrust::raw_pointer_cast(pillars_builder_cuda.GetPillarFeatureData().data())),
        reinterpret_cast<int *>(thrust::raw_pointer_cast(pillars_builder_cuda.GetPillarIndex().data())),
        pillars_builder_cuda.GetPillarIndex().size());
}
