#include "mlp.hpp"
#include "feedforward.hpp"
#include "minimize.hpp"
#include "objectives.hpp"
#include "utilities.hpp"
#include "serialize.hpp"

#include <boost/program_options.hpp>
#include <glog/logging.h>

#include <iostream>
#include <fstream>
#include <functional>
#include <string>
#include <sstream>
#include <utility>
#include <vector>


namespace {
constexpr char VERSION[]{"0.0.3"};

// Name of command line arguments:
constexpr const char* ARG_MODEL{"model"};
constexpr const char* ARG_ARCH{"arch"};
constexpr const char* ARG_LOAD{"load"};
constexpr const char* ARG_SAVE{"save"};
constexpr const char* ARG_LOAD_JSON{"load-json"};
constexpr const char* ARG_SAVE_JSON{"save-json"};
constexpr const char* ARG_EVAL{"eval"};
constexpr const char* ARG_TRAIN{"train"};
constexpr const char* ARG_INPUT_RANGE{"input"};
constexpr const char* ARG_TARGET_RANGE{"target"};
constexpr const char* ARG_TARGET_CATEGORY{"target-category"};
constexpr const char* ARG_N_CATEGORIES{"n-categories"};

constexpr const char* ARG_BATCH_SIZE{"batch-size"};
constexpr const char* ARG_N_BATCHES{"n-batches"};

constexpr const uint32_t DEFAULT_BATCH_SIZE{100};
constexpr const uint32_t DEFAULT_N_BATCHES{100};
constexpr const char* DEFAULT_MODEL{"ff"};

std::pair<uint32_t, uint32_t> parse_range_or_die(const std::string& range_str) {
  auto range = unet::parse_range(range_str);
  if (!range) {
    std::cerr << "Error: " << range_str
              << " could not be parsed as a range (START:END)\n";
    exit(1);
  }
  return *range;
}

struct ModelInfo {
  std::string type;
  std::string description;
};

}  // anonymous namespace

int main(int argc, char **argv) {
  namespace po = boost::program_options;
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  std::string model, model_in_path, model_out_path;
  std::string arch;
  std::string train_file;
  std::string input_range_str, target_range_str;
  bool softmax{false};
  uint32_t target_category{0}, n_categories{0};
  uint32_t batch_size{0}, n_batches{0};
  uint32_t n_input{0}, n_output{0};

  std::stringstream desc_stream;
  desc_stream
    << "μnet - deep neural networks trained with SGD or Hessian free optimisation "
    << "(version " << VERSION << ")";
  po::options_description description{desc_stream.str()};
  description.add_options()
    ("help,h", "Prints this help message.")
    (ARG_MODEL, po::value<std::string>(&model)
     ->value_name("MODEL")->default_value(DEFAULT_MODEL),
     "What type of model to use. Supported: ff (feedforward)")
    (ARG_ARCH, po::value<std::string>(&arch)
     ->value_name("ARCH"), "ff | Specify the architecture of the network, "
      "e.g. 748-500-300-100-30-10")
    (ARG_LOAD, po::value<std::string>(&model_in_path)
     ->value_name("FILE"), "Load a model of the specified type.")
    (ARG_SAVE, po::value<std::string>(&model_out_path)
     ->value_name("FILE"), "Save the model to file.")
    (ARG_LOAD_JSON, po::value<std::string>(&model_in_path)
     ->value_name("FILE"), "Load a model of the specified type from json.")
    (ARG_SAVE_JSON, po::value<std::string>(&model_out_path)
     ->value_name("FILE"), "Save the model to file as json.")
    (ARG_EVAL, po::value<bool>(), "Save the model to file as json.")
    (ARG_TRAIN, po::value<std::string>(&train_file)
     ->value_name("FILE"), "Specify a training file.")
    (ARG_INPUT_RANGE, po::value<std::string>(&input_range_str)
     ->value_name("START:END"),
     "Which elements of the data vectors to use as input to the network. "
     "Specified as a 0-indexed range START:END.")
    (ARG_TARGET_RANGE, po::value<std::string>(&target_range_str)
     ->value_name("START:END"),
     "[Regression] Which elements of the data vectors to use as target.")
    (ARG_TARGET_CATEGORY, po::value<uint32_t>(&target_category)
     ->value_name("INDEX"),
     "[Classification] Interpret the target element as a categorical variable "
      "(i.e. an integer counting from 0).")
    (ARG_N_CATEGORIES, po::value<uint32_t>(&n_categories)
     ->value_name("NUM"), (std::string {"[Classification] Use with --"} +
     ARG_TARGET_CATEGORY + "; How many categories there are in total.").c_str())
    (ARG_BATCH_SIZE, po::value<uint32_t>(&batch_size)
     ->value_name("NUM")->default_value(DEFAULT_BATCH_SIZE), "Mini batch size.")
    (ARG_N_BATCHES, po::value<uint32_t>(&n_batches)
     ->value_name("NUM")->default_value(DEFAULT_N_BATCHES),
     "Total number of mini batches.");

  auto variables = po::variables_map{};
  try {
    po::store(po::parse_command_line(argc, argv, description), variables);
    po::notify(variables);
    if (variables.count("help")) {
      std::cerr << description << "\n";
      return 0;
    }
    std::cerr << "Eigen is using " << Eigen::nbThreads() << " threads." << "\n";
#ifdef EIGEN_USE_MKL_ALL
    std::cerr << "MKL is enabled." << "\n";
#else
    std::cerr << "MKL is disabled." << "\n";
#endif

    std::istream* train_in{nullptr};
    std::ifstream train_file_in;
    if (variables.count(ARG_TRAIN)) {
      std::cerr << "Training set: reading from file " << train_file << "\n";
      train_file_in.open(train_file, std::ios::in);
      if (!train_file_in.is_open()) {
        std::cerr << "Error: unable to open the training file.\n";
        exit(1);
      }
      train_in = &train_file_in;
    } else {
      std::cerr << "Training set: reading from stdin (no file was specified).\n";
      train_in = &std::cin;
    }
    CHECK(train_in != nullptr);

    auto input_range = parse_range_or_die(input_range_str);
    unet::RangeSelector input_transform{input_range};
    n_input = input_range.second - input_range.first;

    std::function<unet::Batch()> read_batch;
    if (variables.count(ARG_TARGET_RANGE)) {
      if (variables.count(ARG_TARGET_CATEGORY)) {
        std::cerr << "Error: only one of --" << ARG_TARGET_RANGE << " or --"
                  << ARG_TARGET_CATEGORY << " may be used.\n";
        exit(1);
      }
      auto target_range = parse_range_or_die(target_range_str);
      n_output = target_range.second - target_range.first;
      read_batch = [=] () {
        unet::RangeSelector target_transform{target_range};
        return unet::read_batch(
          *train_in, batch_size, input_transform, target_transform);
      };
    } else if (variables.count(ARG_TARGET_CATEGORY)) {
      if (!variables.count(ARG_N_CATEGORIES)) {
        std::cerr << "Error: the total number of categories needs to be "
                  << "specified  if a one hot encoder is used (--"
                  << ARG_TARGET_CATEGORY << ")\nUse --"
                  << ARG_N_CATEGORIES << " NUM\n";
        exit(1);
      }
      n_output = n_categories;
      softmax = true;
      read_batch = [=] () {
        unet::OneHotEncoder one_hot_encoder{target_category, n_categories};
        return unet::read_batch(
          *train_in, batch_size, input_transform, one_hot_encoder);
      };
    } else {
      std::cerr << "Error: a target needs to be specified using --"
                << ARG_TARGET_RANGE << " or --" << ARG_TARGET_CATEGORY << "\n";
      exit(1);
    }

    unet::FeedForward net;
    if (variables.count(ARG_LOAD) || variables.count(ARG_LOAD_JSON)) {
      LOG(INFO) << "Loading a model of type " << model << " from file "
                << model_in_path;
      std::fstream model_in{model_in_path, std::ios::in | std::ios::binary};
      if (!model_in.is_open()) {
        LOG(ERROR) << "Could not open file to read model. ";
        exit(2);
      }
      if (variables.count(ARG_LOAD)) {
        unet::load_from_binary(model_in, net);
      } else {
        unet::load_from_json(model_in, net);
      }
    } else {
      if (!variables.count(ARG_ARCH)) {
        std::cerr << "Error: the architecture of the network was not specified."
                  << " Use --" << ARG_ARCH;
        exit(1);
      }
      auto layers = unet::arch_from_str(arch);
      if (layers.size() == 0) {
        std::cerr << "Error: the architecture \"" << arch << "\" "
                  << " could not be parsed.\n";
        exit(1);
      }
      net = unet::FeedForward{layers, softmax};
    }
    if (n_input != net.n_input()) {
      LOG(ERROR) << "The network has " << net.n_input() << " inputs != "
                 << n_input << " required.";
      exit(3);
    } else if (n_output != net.n_output()) {
      LOG(ERROR) << "The network has " << net.n_output() << " outputs != "
                 << n_output << " required.";
      exit(3);
    }

    std::stringstream arch_str;
    for (size_t layer = 0; layer < net.n_layers(); layer ++) {
      arch_str << net.layers()[layer];
      if (layer < net.n_layers() - 1) { arch_str << "-"; }
    }

    LOG(INFO) << "Training a MLP with arch = " << arch_str.str()
              << " (output = " << (softmax ? "softmax" : "linear")
              << ")";
    unet::NesterovGD minimize{0.01, 0.9999, 0.8, 0.95, 0.9996, 1};

    double mean_error{-1};
    for (uint32_t n_batch = 0; n_batch < n_batches; ++n_batch) {
      LOG(INFO) << "Starting mini batch number " << n_batch;
      auto batch = read_batch();
      batch.input /= 255.0;

      if (softmax) {
        auto cross_entropy = net.cross_entropy(batch.input, batch.target);
        // unet::CrossEntropy<unet::FeedForward> cross_entropy{
        //   net, batch.input, batch.target};
        if (variables.count(ARG_EVAL)) {
          unet::Accuracy<unet::FeedForward> acc{net, batch.input, batch.target};
          double error{acc(net.weights())};
          mean_error = mean_error * n_batch / (n_batch + 1) + error / (n_batch + 1);
          LOG(INFO) << "Batch error: " << error
                    << " / Average error so far: " << mean_error;
        } else {
          unet::Accuracy<unet::FeedForward> acc{net, batch.input, batch.target};
          double error{acc(net.weights())};
          if (mean_error == -1) {
            mean_error = error;
          } else {
            mean_error = 0.95 * mean_error + 0.05 * error;
          }
          LOG(INFO) << "Batch accuracy: " << error
                    << " / EWMA accuracy: " << mean_error;

          minimize.fit_batch(cross_entropy, net.weights());
        }
      } else {
        auto l2_error = net.l2_error(batch.input, batch.target);
        if (variables.count(ARG_EVAL)) {
          double error{l2_error(net.weights())};
          mean_error = mean_error * n_batch / (n_batch + 1) + error / (n_batch + 1);
          LOG(INFO) << "Batch error: " << error
                    << " / Average error so far: " << mean_error;
        } else {
          minimize.fit_batch(l2_error, net.weights());
        }
      }
    }

    if (variables.count(ARG_SAVE) || variables.count(ARG_SAVE_JSON)) {
      LOG(INFO) << "Saving the model to file " << model_out_path;
      std::fstream model_out{model_out_path, std::ios::out | std::ios::binary};
      if (!model_out.is_open()) {
        LOG(ERROR) << "Could not open file for writing. ";
        exit(2);
      }
      if (variables.count(ARG_SAVE)) {
        unet::save_to_binary(model_out, net);
      } else {
        unet::save_to_json(model_out, net);
      }
    }
  } catch (const po::error& e) {
    std::cerr << e.what() << "\n";
    exit(1);
  } catch(const std::exception& e) {
    LOG(ERROR) << e.what();
    return -1;
  }
  return 0;
}
