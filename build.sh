#!/bin/bash

(cd ./llama_cpp_dir && make)
if [ $? -ne 0 ]; then
    echo "Unable to build static library 'libllama'"
    exit 1
fi

if [ ! -d "./build" ]; then
    mkdir build
fi

if [ $? -ne 0 ]; then
    echo "Unable create build directory!"
    exit 2
fi

cd build && cmake ../llama_cpp_dir && make

if [ $? -ne 0 ]; then
    echo "Unable to build bridge.cpp and link the 'libllama'"
    exit 3
fi

