#!/usr/bin/env bash
set -euo pipefail

root_dir=$(dirname $0)
submodules=(
  "lockfree"
  "align"
  "array"
  "assert"
  "atomic"
  "config"
  "core"
  "integer"
  "iterator"
  "mpl"
  "parameter"
  "predef"
  "static_assert"
  "tuple"
  "type_traits"
  "utility"
  "winapi"
  "concept_check"
  "mp11"
  "conversion"
  "typeof"
  "move"
  "detail"
  "function_types"
  "fusion"
  "optional"
  "smart_ptr"
  "container_hash"
  "io"
  "preprocessor"
  "throw_exception"
)

set -x
cd $root_dir
git submodule sync
git submodule update --init

# for boost 1.74.0
cd csrc/third_party/boost/libs
git submodule sync
git submodule update --init --jobs 16 "${submodules[@]}"
