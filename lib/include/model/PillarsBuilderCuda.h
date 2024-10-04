#pragma once

#include "Config.hpp"
#include <memory>

class PillarsBuilderCudaImpl;

class PillarsBuilderCuda
{
private:
    PillarsBuilderCudaImpl *pimpl;

public:
    PillarsBuilderCuda();
    ~PillarsBuilderCuda();

    /**
     * @brief Build pillars feature from point cloud data
     *
     * @param data point cloud data
     * @param point_num number of points
     * @return 显存中pillar指针，index指针，有效pillar数目
     */
    std::tuple<float *, int *, int> BuildPillarsFeature(const float *data, int point_num);
};
