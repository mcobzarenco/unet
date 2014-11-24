#include "layer.hpp"
#include "utilities.hpp"

#include <boost/program_options.hpp>
#include <glog/logging.h>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>


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

constexpr const char* ARG_BATCH_SIZE{"batch-size"};

constexpr const int32_t DEFAULT_BATCH_SIZE{100};

}  // anonymous namespace

int main(int argc, char **argv) {
  namespace po = boost::program_options;
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  string model;
  string train_file;
  string input_range_str, target_range_str;
  uint32_t target_category{0}, batch_size{0};
  uint32_t n_input{0}, n_hidden{0}, n_output{0};

  stringstream desc_stream;
  desc_stream
    << "unet - deep neural networks trained with Hessian free optimisation "
    << "(version " << VERSION << ")";
  po::options_description description{desc_stream.str()};
  description.add_options()
    ("help,h", "Prints this help message.")
    (ARG_MODEL, po::value<string>(&model)
     ->value_name("MODEL"), "What model to use: mlp")
    (ARG_MLP_N_HIDDEN, po::value<uint32_t>(&n_hidden)
     ->value_name("NUM"), "For MLPs | Number of hidden units.")
    (ARG_TRAIN, po::value<string>(&train_file)
     ->value_name("FILE"),
     "Specify a training file.")
    (ARG_INPUT_RANGE, po::value<string>(&input_range_str)
     ->value_name("START:END"),
     "Which elements of the data vectors to use as input to the network. "
     "Specified as a 0-indexed range START:END.")
    (ARG_TARGET_RANGE, po::value<string>(&target_range_str)
     ->value_name("START:END"),
     "[Regression] Which elements of the data vectors to use as target.")
    (ARG_TARGET_CATEGORY, po::value<uint32_t>(&target_category)
     ->value_name("INDEX"),
     "[Classification] Interpret the target element as a categorical variable.")
    (ARG_BATCH_SIZE, po::value<uint32_t>(&batch_size)
     ->value_name("N")->default_value(DEFAULT_BATCH_SIZE), "Mini batch size.");

  auto variables = po::variables_map{};
  try {
    po::store(po::parse_command_line(argc, argv, description), variables);
    po::notify(variables);
    if (variables.count("help")) {
      cerr << description << endl;
      return 0;
    }

    if (!variables.count(ARG_MLP_N_HIDDEN)) {
      cerr << "The number of units in the hidden layer needs to be defined.\n"
           << "Use --" << ARG_MLP_N_HIDDEN << " NUMBER\n\n"
           << "Exiting.\n";
      exit(1);
    }

    istream* train_in{nullptr};
    ifstream train_file_in;
    if (variables.count(ARG_TRAIN)) {
      LOG(INFO) << "Reading training data from file: " << train_file;
      train_file_in.open(train_file, ios::in);
      train_in = &train_file_in;
    } else {
      LOG(INFO) << "Reading training data from stdin (no file was specified).";
      train_in = &cin;
    }
    CHECK(train_in != nullptr);

    auto input_range = *unet::parse_range(input_range_str);
    // auto target_range = *unet::parse_range(target_range_str);

    n_input = input_range.second - input_range.first;
    // n_output = target_range.second - target_range.first;

    unet::RangeSelector input_transform{input_range};
    // unet::RangeSelector target_transform{target_range};
    unet::OneHotEncoder target_transform{0, 10};
    n_output = 10;

    auto batch = unet::read_batch(
      *train_in, batch_size, input_transform, target_transform);

    // cerr << "batch_X=\n"<< batch.input << endl;
    // cerr << "batch_Y=\n"<< batch.target << endl;

    unet::MLP mlp(batch.n_input, n_hidden, batch.n_output, true);
    mlp.l2_error(batch.input, batch.target).minimize_gd(3000);
  } catch (const boost::program_options::unknown_option& e) {
    LOG(ERROR) << e.what();
    return 1;
  } catch (const boost::program_options::invalid_option_value& e) {
    LOG(ERROR) << e.what();
    return 2;
  } catch(const std::exception& e) {
    LOG(ERROR) << e.what();
    return -1;
  }
  // return 0;

  // std::random_device rd;
  // std::mt19937 generator{rd()};
  // std::normal_distribution<> normal(0, .1);

  // unet::MLP net(2, 3, 1, false, [&] () {return normal(generator);});

  Matrix<double, Dynamic, Dynamic> x{2, 5}, y{2, 1};
  Eigen::Map<Eigen::VectorXd> y_map{y.data(), 2};
  x <<
    0, 0, 1, 1, .9,
    0, 1, 0, 1, .1;
  y << 3, 1; //, 1, 0, .9;

  x.colwise() += y_map;
  cout << "x" << x << "\n";

  // auto l2_error = net.l2_error(x, y);
  // l2_error.minimize_gd(1000);
}
