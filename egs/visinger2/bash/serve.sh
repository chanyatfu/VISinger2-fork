
exp_dir=$(pwd)
base_dir=$(dirname $(dirname $exp_dir))

export PYTHONPATH=$base_dir
export PYTHONIOENCODING=UTF-8

CUDA_VISIBLE_DEVICES=0 python serve.py \
    -model_dir ../../models/opencpop
