# Convert MMS to Sherpa-ONNX models and push to huggingface

**See converted onnx models at https://huggingface.co/willwade/mms-tts-multilingual-models-onnx/tree/main **

## Setup

1. Create a repository on Hugging Face (you can do this via the Hugging Face website or CLI).
2. Generate an API token from your Hugging Face account settings.

3. Build regular Sherpa-ONNX - this is for a GPU machine - NB: YOU NEED TO INSTALL CUDA FIRST. Follow kinda the instructions at https://k2-fsa.github.io/k2/installation/cuda-cudnn.html

```bash
cd ~
sudo apt-get install alsa-utils libasound2-dev
git clone https://github.com/k2-fsa/sherpa-onnx
cd sherpa-onnx
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DSHERPA_ONNX_ENABLE_GPU=ON ..
make -j6
```

4. Get ready for our code. 

```bash
cd ~
git clone https://github.com/willwade/mms-tts-multilingual-models-onnx
cd mms-tts-multilingual-models-onnx
pip install -qq torch==1.13.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

git clone https://huggingface.co/spaces/mms-meta/MMS
export PYTHONPATH=$PWD/MMS:$PYTHONPATH
export PYTHONPATH=$PWD/MMS/vits:$PYTHONPATH

pushd MMS/vits/monotonic_align

python3 setup.py build

ls -lh build/
ls -lh build/lib*/
ls -lh build/lib*/*/

cp build/lib*/vits/monotonic_align/core*.so .

sed -i.bak s/.monotonic_align.core/.core/g ./__init__.py
popd


export HF_TOKEN=your_huggingface_token
```

## Run

```bash
python main.py
```
