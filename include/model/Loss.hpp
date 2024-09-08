#pragma once
#include <unordered_map>
#include <torch/torch.h>
#include "../Datatype.hpp"
class Loss
{
private:
    /* data */
public:
    Loss(/* args */)
    {

    }

    torch::Tensor operator()(std::unordered_map<std::string, torch::Tensor> pred, const std::vector<Object> objs_ground)
    {
        
    }
};
