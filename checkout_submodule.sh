#!/usr/bin/env bash
root_dir=$(dirname $0)
set -x
cd $root_dir
git submodule update --init
# for boost 1.74.0
git -C csrc/third_party/boost/libs submodule update --init --jobs 16\
    lockfree align array assert atomic config core integer iterator mpl\
    parameter predef static_assert tuple type_traits utility winapi\
    concept_check mp11 conversion typeof move detail function_types fusion\
    optional smart_ptr container_hash io preprocessor throw_exception
