#pragma once

#include "model/Loss.hpp"
#include "model/Model.hpp"
#include "TicToc.hpp"

class Detector
{
private:
    using CloudType = pcl::PointCloud<PillarsBuilder::PointType>;
    using TensorMap = std::unordered_map<std::string, torch::Tensor>;
    using DataPair = std::pair<pcl::PointCloud<PillarsBuilder::PointType>, std::vector<Object>>;

public:
    enum class Mode
    {
        TRAIN,
        INFERENCE
    };

    Detector(Mode mode = Mode::TRAIN) : mode_(mode)
    {
        if (mode == Mode::TRAIN)
            model->train();
        else
            model->eval();

        optimizer = std::make_unique<torch::optim::Adam>(model->parameters(), torch::optim::AdamOptions(1e-6));
    }

    void Train(CloudType &cloud, const std::vector<Object> &objs_gt);
    void Train(const std::vector<DataPair> &data, int accumulation_steps = 1);
    void Infer(CloudType &cloud, std::vector<Object> &objs_infer, float theshold = 0.3);
    void SaveModeParamters(const std::string &path);
    void LoadModeParamters(const std::string &path);

private:
    DetectNet model;
    Loss loss_function;
    Mode mode_;

    std::vector<Object> objs_infer;
    std::unique_ptr<torch::optim::Adam> optimizer;
    torch::Device device = torch::Device(torch::kCUDA);

    void BuildDetectionGroundTruth(const std::vector<Object> &objs, TensorMap &ground_truth);
    void BuildDetectionGroundTruth(const std::vector<DataPair> &data, TensorMap &ground_truth);
};