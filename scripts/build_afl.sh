#!/bin/bash

echo "Initializing and updating submodule: AFL"
git submodule update --init --recursive

echo "Successfully updated, building afl now ..."
# shellcheck disable=SC2164
cd afl/

make
echo "AFL successfully built"