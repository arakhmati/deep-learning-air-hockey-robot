#!/bin/bash
rm data/* mixed_data/*
./generate_data.sh
./mix_random_images_with_data.py
