#include "mlp.hpp"
#include "minimize.hpp"
#include "objectives.hpp"
#include "utilities.hpp"

#include <boost/program_options.hpp>
#include <glog/logging.h>

#include <iostream>
#include <fstream>
#include <functional>
#include <string>
#include <sstream>
#include <utility>


using namespace std;
using Eigen::Matrix;
using Eigen::Dynamic;

namespace {
constexpr char VERSION[]{"0.0.1"};

// Name of command line arguments:
constexpr const char* ARG_MODEL{"model"};
constexpr const char* ARG_MLP_N_HIDDEN{"n-hidden"};
constexpr const char* ARG_TRAIN{"train"};
constexpr const char* ARG_INPUT_RANGE{"input"};
constexpr const char* ARG_TARGET_RANGE{"target"};
constexpr const char* ARG_TARGET_CATEGORY{"target-category"};
constexpr const char* ARG_N_CATEGORIES{"n-categories"};

constexpr const char* ARG_BATCH_SIZE{"batch-size"};

constexpr const int32_t DEFAULT_BATCH_SIZE{100};
constexpr const char* DEFAULT_MODEL{"mlp"};

}  // anonymous namespace


std::pair<uint32_t, uint32_t> parse_range_or_die(const std::string& range_str) {
  auto range = unet::parse_range(range_str);
  if (!range) {
    cerr << "Error: " << range_str
         << " could not be parsed as a range (START:END)\n";
    exit(1);
  }
  return *range;
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  string model;
  string train_file;
  string input_range_str, target_range_str;
  bool softmax{false};
  uint32_t target_category{0}, n_categories{0};
  uint32_t batch_size{0};
  uint32_t n_input{0}, n_hidden{0}, n_output{0};

  stringstream desc_stream;
  desc_stream
    << "μnet - deep neural networks trained with Hessian free optimisation "
    << "(version " << VERSION << ")";
  po::options_description description{desc_stream.str()};
  description.add_options()
    ("help,h", "Prints this help message.")
    (ARG_MODEL, po::value<string>(&model)
     ->value_name("MODEL")->default_value(DEFAULT_MODEL),
     "What model to use: mlp")
    (ARG_MLP_N_HIDDEN, po::value<uint32_t>(&n_hidden)
     ->value_name("NUM"), "For MLPs | Number of hidden units.")
    (ARG_TRAIN, po::value<string>(&train_file)
     ->value_name("FILE"), "Specify a training file.")
    (ARG_INPUT_RANGE, po::value<string>(&input_range_str)
     ->value_name("START:END"),
     "Which elements of the data vectors to use as input to the network. "
     "Specified as a 0-indexed range START:END.")
    (ARG_TARGET_RANGE, po::value<string>(&target_range_str)
     ->value_name("START:END"),
     "[Regression] Which elements of the data vectors to use as target.")
    (ARG_TARGET_CATEGORY, po::value<uint32_t>(&target_category)
     ->value_name("INDEX"),
     "[Classification] Interpret the target element as a categorical variable "
      "(i.e. an integer counting from 0).")
    (ARG_N_CATEGORIES, po::value<uint32_t>(&n_categories)
     ->value_name("NUM"), (string {"[Classification] Use with --"} +
     ARG_N_CATEGORIES + "; How many categories there are in total.").c_str())
    (ARG_BATCH_SIZE, po::value<uint32_t>(&batch_size)
     ->value_name("NUM")->default_value(DEFAULT_BATCH_SIZE), "Mini batch size.");

  auto variables = po::variables_map{};
  try {
    po::store(po::parse_command_line(argc, argv, description), variables);
    po::notify(variables);
    if (variables.count("help")) {
      cerr << description << endl;
      return 0;
    }
    cerr << "Eigen is using " << Eigen::nbThreads() << " threads." << "\n";
#ifdef EIGEN_USE_MKL_ALL
    cerr << "MKL is enabled." << "\n";
#else
    cerr << "MKL is disabled." << "\n";
#endif

    istream* train_in{nullptr};
    ifstream train_file_in;
    if (variables.count(ARG_TRAIN)) {
      cerr << "Training set: reading from file " << train_file << "\n";
      train_file_in.open(train_file, ios::in);
      if (!train_file_in.is_open()) {
        cerr << "Error: unable to open the training file.\n";
        exit(1);
      }
      train_in = &train_file_in;
    } else {
      cerr << "Training set: reading from stdin (no file was specified).\n";
      train_in = &cin;
    }
    CHECK(train_in != nullptr);

    auto input_range = parse_range_or_die(input_range_str);
    unet::RangeSelector input_transform{input_range};
    n_input = input_range.second - input_range.first;

    std::function<unet::Batch()> read_batch;
    if (variables.count(ARG_TARGET_RANGE)) {
      if (variables.count(ARG_TARGET_CATEGORY)) {
        cerr << "Error: only one of --" << ARG_TARGET_RANGE << " or --"
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
        cerr << "Error: the total number of categories needs to be specified "
             << "if a one hot encoder is used (--" << ARG_TARGET_CATEGORY
             << ")\nUse --" << ARG_N_CATEGORIES << " NUM\n";
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
      cerr << "Error: a target needs to be specified using --"
           << ARG_TARGET_RANGE << " or --" << ARG_TARGET_CATEGORY << "\n";
      exit(1);
    }

    if (!variables.count(ARG_MLP_N_HIDDEN)) {
      cerr << "Error: the number of units in the hidden layer needs to be defined.\n"
           << "Use --" << ARG_MLP_N_HIDDEN << " NUMBER\n";
      exit(1);
    }

    // cerr << "batch_X=\n"<< batch.input << endl;
    // cerr << "batch_Y=\n"<< batch.target << endl;
    LOG(INFO) << "Training a MLP with arch = " << n_input << " -> "
              << n_hidden << " -> " << n_output
              << " (output = " << (softmax ? "softmax" : "linear")
              << ")";
    unet::MLP mlp{n_input, n_hidden, n_output, softmax};
    unet::MomentumGD minimize{0.01, 0.8, 5, 1.03, 0.8, 0.999};

    for (int i = 0; i < 500; ++i) {
      LOG(INFO) << "Starting batch number " << i;
      auto batch = read_batch();
      // batch.input /= 100.0;
      unet::L2Error<unet::MLP> l2_error{mlp, batch.input, batch.target};
      minimize.fit_batch(l2_error);
    }
  } catch (const boost::program_options::error& e) {
    cerr << e.what() << "\n";
    exit(1);
  } catch(const std::exception& e) {
    LOG(ERROR) << e.what();
    return -1;
  }
  return 0;
}
