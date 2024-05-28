# Ensure OPENVINO_SRC_DIR is set
if [[ -z "$OPENVINO_SRC_DIR" ]]; then
    echo "Please export OPENVINO_SRC_DIR=/path/to/cloned/repo (no trailing /) before running this script" 1>&2
    exit 1
fi

# Setup python environment and install python build dependencies
if [ ! -d ".venv" ]; then
    python3.11 -m venv ./.venv
fi
source ./.venv/bin/activate
pip install -U pip
pip install -r ./src/bindings/python/wheel/requirements-dev.txt

# Build with python whl
rm -rf build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTHON=ON -DPython3_EXECUTABLE=/usr/bin/python3.11 -DENABLE_WHEEL=ON ..
cmake --build . --parallel 24

# Build C++ samples
cd ../samples/cpp
source ../../scripts/setupvars/setupvars.sh
export OpenVINO_DIR="${OPENVINO_SRC_DIR}/build"
./build_samples.sh

# Prepare environment to run python script
source ../../.venv/bin/activate
pip install transformers optimum[intel] nncf tqdm
find ../../build/wheels -name '*manylinux*.whl' -exec pip install {} \;
