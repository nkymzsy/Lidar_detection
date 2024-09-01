#pragma once

#include <Eigen/Dense>

struct Object
{
    int lable;
    float heading;
    Eigen::Vector3f position;
    Eigen::Vector3f dimensions;

    Object(int l, float h, Eigen::Vector3f p, Eigen::Vector3f d) : lable(l), position(p), dimensions(d), heading(h) {}
};