#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <vector>
#include <cstdint>

struct ModelConfig {
    int min_ngram;
    int max_ngram;
    int feature_dimensions;
    float min_confidence;
};

struct Prediction {
    std::string name;
    float girl_prob;
    float boy_prob;
    float uncertain_prob;
    std::string classification;
    std::string confidence;
};

class GenderFluidModel {
public:
    GenderFluidModel();
    ~GenderFluidModel();

    bool load(const std::string& path);
    Prediction predict(const std::string& name) const;
    std::vector<Prediction> predict_batch(const std::vector<std::string>& names) const;
    const ModelConfig& config() const { return config_; }
    size_t model_size() const { return model_size_; }

private:
    std::string normalize_name(const std::string& name) const;
    std::vector<float> extract_features(const std::string& name) const;
    std::vector<float> softmax(const std::vector<float>& logits) const;

    ModelConfig config_;
    std::vector<std::vector<float>> coef_;  // [3][feature_dims]
    std::vector<float> intercept_;          // [3]
    size_t model_size_;
    bool loaded_;
};

#endif // MODEL_H
