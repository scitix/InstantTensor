#!/usr/bin/env bash
set -euo pipefail

root_dir=$(dirname $0)

set -x
cd $root_dir
git submodule sync
git submodule update --init
