#pragma once

#include <Eigen/Dense>

/**
 * @brief 表示一个对象的信息，包括类别标签、航向角、位置和尺寸。
 */
struct Object
{
    int label;                  // 标签，用于标识对象的类别
    float heading;              // 航向角，表示对象的方向
    Eigen::Vector3f position;   // 三维位置坐标
    Eigen::Vector3f dimensions; // 三维尺寸

    /**
     * @brief 构造函数，初始化对象的属性。
     *
     * @param l 标签
     * @param h 航向角
     * @param p 位置坐标
     * @param d 尺寸
     */
    Object(int l, float h, const Eigen::Vector3f &p, const Eigen::Vector3f &d)
        : label(l), heading(h), position(p), dimensions(d)
    {
    }
};