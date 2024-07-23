# Convert MMS to Sherpa-ONNX models and push to huggingface


## Setup

```bash
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
```

## Run

```bash
python main.py
```