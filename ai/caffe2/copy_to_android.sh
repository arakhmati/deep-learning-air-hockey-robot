#!/usr/bin/env bash

./convert_keras_to_caffe2.py
cp *.pb ../../../perception/app/src/main/assets
