#include <string>

#include <algorithm>
#define PROTOBUF_USE_DLLS 1
#define CAFFE2_USE_LITE_PROTO 1
#include <caffe2/core/predictor.h>
#include <caffe2/core/operator.h>
#include <caffe2/core/timer.h>

#include "caffe2/core/init.h"
#include "caffe2/core/flags.h"
#include "caffe2/core/init.h"
#include "caffe2/core/predictor.h"
#include "caffe2/utils/proto_utils.h"

#define IMG_H 128
#define IMG_W 128
#define IMG_C 3
#define IMG_D 9 // Depth
#define MAX_DATA_SIZE IMG_H * IMG_W * IMG_D

static caffe2::NetDef _initNet, _predictNet;
static caffe2::Predictor *_predictor;
static char raw_data[MAX_DATA_SIZE];
static float input_data[MAX_DATA_SIZE];
static caffe2::Workspace ws;

const char * directions_map[] {
        "NW", "N", "NE",
        "W", "", "E",
        "SW", "S", "SE", "Invalid"
};

void initCaffe2() {
    CAFFE_ENFORCE(ReadProtoFromFile("init_net.pb", &_initNet));
    CAFFE_ENFORCE(ReadProtoFromFile("predict_net.pb", &_predictNet));
}

float avg_fps = 0.0;
float total_fps = 0.0;
int iters_fps = 10;


void classificationFromCaffe2(
        int h, int w, char* Y, char* U, char* V,
        int rowStride, int pixelStride,
        char infer_HWC) {

    int Y_len = 0;
    char * Y_data = Y;
    assert(Y_len <= MAX_DATA_SIZE);
    int U_len = 0;
    char * U_data = U;
    assert(U_len <= MAX_DATA_SIZE);
    int V_len = 0;
    char * V_data = V;
    assert(V_len <= MAX_DATA_SIZE);

#define min(a,b) ((a) > (b)) ? (b) : (a)
#define max(a,b) ((a) > (b)) ? (a) : (b)

    auto h_offset = max(0, (h - IMG_H) / 2);
    auto w_offset = max(0, (w - IMG_W) / 2);

    auto iter_h = IMG_H;
    auto iter_w = IMG_W;
    if (h < IMG_H) {
        iter_h = h;
    }
    if (w < IMG_W) {
        iter_w = w;
    }

    for (auto i = 0; i < iter_h; ++i) {
        char* Y_row = &Y_data[(h_offset + i) * w];
        char* U_row = &U_data[(h_offset + i) / 4 * rowStride];
        char* V_row = &V_data[(h_offset + i) / 4 * rowStride];
        for (auto j = 0; j < iter_w; ++j) {
            // Tested on Pixel and S7.
            char y = Y_row[w_offset + j];
            char u = U_row[pixelStride * ((w_offset+j)/pixelStride)];
            char v = V_row[pixelStride * ((w_offset+j)/pixelStride)];

            float b_mean = 104.00698793f;
            float g_mean = 116.66876762f;
            float r_mean = 122.67891434f;

            auto b_i = 0 * IMG_H * IMG_W + j * IMG_W + i;
            auto g_i = 1 * IMG_H * IMG_W + j * IMG_W + i;
            auto r_i = 2 * IMG_H * IMG_W + j * IMG_W + i;

            auto b_i_1 = b_i + (IMG_H * IMG_W * IMG_C);
            auto g_i_1 = g_i + (IMG_H * IMG_W * IMG_C);
            auto r_i_1 = r_i + (IMG_H * IMG_W * IMG_C);
//
            auto b_i_2 = b_i + 2 * (IMG_H * IMG_W * IMG_C);
            auto g_i_2 = g_i + 2 * (IMG_H * IMG_W * IMG_C);
            auto r_i_2 = r_i + 2 * (IMG_H * IMG_W * IMG_C);


//            if (infer_HWC) {
//                b_i = (j * IMG_W + i) * IMG_C;
//                g_i = (j * IMG_W + i) * IMG_C + 1;
//                r_i = (j * IMG_W + i) * IMG_C + 2;
//            }
/*
  R = Y + 1.402 (V-128)
  G = Y - 0.34414 (U-128) - 0.71414 (V-128)
  B = Y + 1.772 (U-V)
 */

//            input_data[r_i_2] = input_data[r_i_1];
//            input_data[g_i_2] = input_data[g_i_1];
//            input_data[b_i_2] = input_data[b_i_1];
//
//            input_data[r_i_1] = input_data[r_i];
//            input_data[g_i_1] = input_data[g_i];
//            input_data[b_i_1] = input_data[b_i];

            input_data[r_i] = -r_mean + (float) ((float) min(255., max(0., (float) (y + 1.402 * (v - 128)))));
            input_data[g_i] = -g_mean + (float) ((float) min(255., max(0., (float) (y - 0.34414 * (u - 128) - 0.71414 * (v - 128)))));
            input_data[b_i] = -b_mean + (float) ((float) min(255., max(0., (float) (y + 1.772 * (u - v)))));

//            input_data[r_i] = (input_data[r_i] + 128) / 255;
//            input_data[g_i] = (input_data[g_i] + 128) / 255;
//            input_data[b_i] = (input_data[b_i] + 128) / 255;


        }
    }
//    alog("Exited for-loop")
    caffe2::TensorCPU input;
    input.Resize(std::vector<int>({1, IMG_D, IMG_H, IMG_W}));

    memcpy(input.mutable_data<float>(), input_data, IMG_H * IMG_W * IMG_D * sizeof(float));
    caffe2::Predictor::TensorVector input_vec{&input};
    caffe2::Predictor::TensorVector output_vec;
    caffe2::Timer t;
    t.Start();
    _predictor->run(input_vec, &output_vec);

    float fps = 1000/t.MilliSeconds();
    total_fps += fps;
    avg_fps = total_fps / iters_fps;
    total_fps -= avg_fps;

    for (auto output : output_vec) {
        for (auto i = 0; i < output->size(); ++i) {
            printf("%f", output->template data<float>()[i]);
        }
    }

    constexpr int k = 5;
    float max[k] = {0};
    int max_index[k] = {0};
    // Find the top-k results manually.
    if (output_vec.capacity() > 0) {
        for (auto output : output_vec) {
            for (auto i = 0; i < output->size(); ++i) {
                for (auto j = 0; j < k; ++j) {
                    if (output->template data<float>()[i] > max[j]) {
                        for (auto _j = k - 1; _j > j; --_j) {
                            max[_j - 1] = max[_j];
                            max_index[_j - 1] = max_index[_j];
                        }
                        max[j] = output->template data<float>()[i];
                        max_index[j] = i;
                        goto skip;
                    }
                }
                skip:;
            }
        }
    }
    std::ostringstream stringStream;
    stringStream << avg_fps << " FPS\n";

    for (auto j = 0; j < k; ++j) {
        stringStream << j << ": " << directions_map[max_index[j]] << " - " << max[j] * 100 << "%\n";
    }
}

int main() {
    
    initCaffe2();
    return 0;
}
