#include "model.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstring>

static const uint32_t MAGIC = 0x00544647; // "GFT\0" little-endian
static const uint32_t FORMAT_VERSION = 1;

GenderFluidModel::GenderFluidModel()
    : model_size_(0), loaded_(false) {
    config_ = {2, 5, 4096, 0.70f};
}

GenderFluidModel::~GenderFluidModel() {}

std::string GenderFluidModel::normalize_name(const std::string& name) const {
    std::string result;
    result.reserve(name.size());

    for (char c : name) {
        if (c == '-' || c == '\'' ) {
            result += ' ';
        } else if (std::isalpha(static_cast<unsigned char>(c))) {
            result += std::tolower(static_cast<unsigned char>(c));
        } else if (c == ' ') {
            result += ' ';
        }
        // skip other punctuation, preserve accented chars as-is
        else if (static_cast<unsigned char>(c) > 127) {
            result += c;
        }
    }

    // collapse whitespace
    std::string collapsed;
    bool prev_space = false;
    for (char c : result) {
        if (c == ' ') {
            if (!prev_space) collapsed += ' ';
            prev_space = true;
        } else {
            collapsed += c;
            prev_space = false;
        }
    }

    // trim
    size_t start = collapsed.find_first_not_of(' ');
    if (start == std::string::npos) return "";
    size_t end = collapsed.find_last_not_of(' ');
    return collapsed.substr(start, end - start + 1);
}

static int hash_ngram(const std::string& ngram, int dimensions) {
    uint64_t h = 0;
    for (char c : ngram) {
        h = h * 31 + static_cast<unsigned char>(c);
    }
    return static_cast<int>(h % dimensions);
}

static int sign_hash(const std::string& ngram) {
    uint64_t h = 0;
    for (char c : ngram) {
        h = h * 131 + static_cast<unsigned char>(c);
    }
    return (h % 2 == 0) ? 1 : -1;
}

std::vector<float> GenderFluidModel::extract_features(const std::string& name) const {
    int dims = config_.feature_dimensions;
    std::vector<float> features(dims, 0.0f);

    for (int n = config_.min_ngram; n <= config_.max_ngram; ++n) {
        std::string padded(n - 1, ' ');
        padded += name;
        padded += std::string(n - 1, ' ');

        for (size_t i = 0; i + n <= padded.size(); ++i) {
            std::string ngram = padded.substr(i, n);
            int idx = hash_ngram(ngram, dims);
            int sign = sign_hash(ngram);
            features[idx] += sign;
        }
    }

    // L2 normalize
    float norm = 0.0f;
    for (float v : features) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 0) {
        for (float& v : features) v /= norm;
    }

    return features;
}

std::vector<float> GenderFluidModel::softmax(const std::vector<float>& logits) const {
    float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    std::vector<float> probs(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }
    for (float& p : probs) p /= sum;
    return probs;
}

bool GenderFluidModel::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;

    // Read magic
    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != MAGIC) return false;

    // Read version
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), 4);
    if (version != FORMAT_VERSION) return false;

    // Read config JSON length + config (skip parsing for simplicity)
    uint32_t config_len;
    file.read(reinterpret_cast<char*>(&config_len), 4);
    file.seekg(config_len, std::ios::cur);

    // Read coefficients shape
    int32_t rows, cols;
    file.read(reinterpret_cast<char*>(&rows), 4);
    file.read(reinterpret_cast<char*>(&cols), 4);

    config_.feature_dimensions = cols;
    coef_.resize(rows, std::vector<float>(cols));
    for (int i = 0; i < rows; ++i) {
        file.read(reinterpret_cast<char*>(coef_[i].data()), cols * 4);
    }

    // Read intercept
    int32_t int_len;
    file.read(reinterpret_cast<char*>(&int_len), 4);
    intercept_.resize(int_len);
    file.read(reinterpret_cast<char*>(intercept_.data()), int_len * 4);

    model_size_ = static_cast<size_t>(file.tellg());
    loaded_ = true;
    return true;
}

Prediction GenderFluidModel::predict(const std::string& name) const {
    Prediction pred;
    pred.name = name;

    std::string normalized = normalize_name(name);
    if (normalized.empty()) {
        pred.girl_prob = 0.33f;
        pred.boy_prob = 0.33f;
        pred.uncertain_prob = 0.34f;
        pred.classification = "uncertain";
        pred.confidence = "low";
        return pred;
    }

    auto features = extract_features(normalized);

    // Linear model: logits = coef @ features + intercept
    std::vector<float> logits(intercept_.size(), 0.0f);
    for (size_t i = 0; i < coef_.size(); ++i) {
        float dot = 0.0f;
        for (size_t j = 0; j < features.size() && j < coef_[i].size(); ++j) {
            dot += coef_[i][j] * features[j];
        }
        logits[i] = dot + intercept_[i];
    }

    auto probs = softmax(logits);

    pred.girl_prob = probs[0];
    pred.boy_prob = probs[1];
    pred.uncertain_prob = probs.size() > 2 ? probs[2] : 0.0f;

    float max_prob = std::max({pred.girl_prob, pred.boy_prob, pred.uncertain_prob});
    if (max_prob < config_.min_confidence) {
        pred.classification = "uncertain";
    } else if (pred.girl_prob == max_prob) {
        pred.classification = "girl-associated";
    } else if (pred.boy_prob == max_prob) {
        pred.classification = "boy-associated";
    } else {
        pred.classification = "uncertain";
    }

    if (max_prob >= 0.90f) pred.confidence = "high";
    else if (max_prob >= 0.70f) pred.confidence = "medium";
    else pred.confidence = "low";

    return pred;
}

std::vector<Prediction> GenderFluidModel::predict_batch(const std::vector<std::string>& names) const {
    std::vector<Prediction> results;
    results.reserve(names.size());
    for (const auto& name : names) {
        results.push_back(predict(name));
    }
    return results;
}
