/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "caffe2/core/init.h"
#include "caffe2/core/flags.h"
#include "caffe2/core/predictor.h"
#include "caffe2/utils/proto_utils.h"

#include <caffe2/core/operator.h>
#include <caffe2/core/operator_gradient.h>



CAFFE2_DEFINE_string(init_net, "", "The given path to the init protobuffer.");
CAFFE2_DEFINE_string(
    predict_net,
    "",
    "The given path to the predict protobuffer.");

#define IMG_D 9
#define IMG_H 128
#define IMG_W 128
#define MAX_DATA_SIZE IMG_H * IMG_W * IMG_D

static float input_data[MAX_DATA_SIZE];


namespace caffe2 {

  void print(const Blob *blob, const std::string &name) {
    auto tensor = blob->Get<TensorCPU>();
    const auto &data = tensor.data<float>();
    std::cout << name << "(" << tensor.dims()
              << "): " << std::vector<float>(data, data + tensor.size())
              << std::endl;
  }

  void run() {
    if (FLAGS_init_net.empty()) {
      LOG(FATAL) << "No init net specified. Use --init_net=/path/to/net.";
    }
    if (FLAGS_predict_net.empty()) {
      LOG(FATAL) << "No predict net specified. Use --predict_net=/path/to/net.";
    }

    caffe2::NetDef init_net, predict_net;
    std::cout << "Loading init_net\n";
    CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_init_net, &init_net));
    std::cout << "Loading predict_net\n";
    CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_predict_net, &predict_net));

    Workspace workspace;
    CAFFE_ENFORCE(workspace.RunNetOnce(init_net));
    CAFFE_ENFORCE(workspace.CreateNet(predict_net));

    // for (auto blob: workspace.Blobs())
    //   std::cout << blob << std::endl;

    std::vector<float> frame(9 * 128 * 128);
    for (auto& v : frame) {
      v = (float)rand() / RAND_MAX;
    }

    auto tensor = workspace.GetBlob("in")->GetMutable<TensorCPU>();
    auto value = TensorCPU({1, 9, 128, 128}, frame, NULL);
    tensor->ResizeLike(value);
    tensor->ShareData(value);

    CAFFE_ENFORCE(workspace.RunNet(predict_net.name()));

    // Print weights of fthe irst two layers
    print(workspace.GetBlob("conv1_w"), "conv1_w");
    print(workspace.GetBlob("conv1_b"), "conv1_b");

    print(workspace.GetBlob("batchnorm1_s"),   "batchnorm1_s");
    print(workspace.GetBlob("batchnorm1_b"),   "batchnorm1_b");
    print(workspace.GetBlob("batchnorm1_rm"),  "batchnorm1_rm");
    print(workspace.GetBlob("batchnorm1_riv"), "batchnorm1_riv");

    // Print input to the network
    print(workspace.GetBlob("in"), "in");

    // Print outputs of the first three layers
    print(workspace.GetBlob("conv1"), "conv1");
    print(workspace.GetBlob("batchnorm1"), "batchnorm1");
    print(workspace.GetBlob("pool1"), "pool1");

    // Print output of the neural network
    print(workspace.GetBlob("softmax"), "softmax");
    
  }
}

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  // This is to allow us to use memory leak checks.
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
