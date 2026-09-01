#include "model.h"
#include <iostream>
#include <string>
#include <cstring>

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [OPTIONS] <name>" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -m <path>    Path to model file" << std::endl;
    std::cerr << "  -j           Output JSON" << std::endl;
    std::cerr << "  -h           Show this help" << std::endl;
}

static void print_prediction(const Prediction& pred, bool json_output = false) {
    if (json_output) {
        std::cout << "{\"name\":\"" << pred.name
                  << "\",\"girl_associated_probability\":" << pred.girl_prob
                  << ",\"boy_associated_probability\":" << pred.boy_prob
                  << ",\"uncertain_probability\":" << pred.uncertain_prob
                  << ",\"classification\":\"" << pred.classification
                  << "\",\"confidence\":\"" << pred.confidence << "\"}" << std::endl;
    } else {
        std::cout << "Name: " << pred.name << std::endl;
        std::cout << std::endl;
        std::cout << "Girl-associated: " << (pred.girl_prob * 100.0f) << "%" << std::endl;
        std::cout << "Boy-associated: " << (pred.boy_prob * 100.0f) << "%" << std::endl;
        std::cout << "Uncertain: " << (pred.uncertain_prob * 100.0f) << "%" << std::endl;
        std::cout << std::endl;
        std::cout << "Classification: " << pred.classification << std::endl;
        std::cout << "Confidence: " << pred.confidence << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::string model_path = "models/genderfluid-tiny.bin";
    bool json_output = false;
    std::string name;

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-m" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::string(argv[i]) == "-j") {
            json_output = true;
        } else if (std::string(argv[i]) == "-h" || std::string(argv[i]) == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (argv[i][0] != '-') {
            name = argv[i];
        }
    }

    if (name.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    GenderFluidModel model;
    if (!model.load(model_path)) {
        std::cerr << "Error: failed to load model from " << model_path << std::endl;
        std::cerr << "Run: python train.py" << std::endl;
        return 1;
    }

    Prediction pred = model.predict(name);
    print_prediction(pred, json_output);

    return 0;
}
