# Installation

```sh
uv venv env_torchtitan --python 3.12
source env_torchtitan/bin/activate
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --force-reinstall
uv pip install --pre torchtitan --index-url https://download.pytorch.org/whl/nightly/cu126
uv pip install -e .
uv pip install debugpy-run
```

# Run code

```sh
git checkout epita-exam
python create_tiny_tokenizer.py 
./run_train.sh
# ./run_train.sh --debug
```

# TODO:

- Faire test suite
    - préparer `my_train.py`
    - script comparison diff loss/grad_norm
    - add determinism to eeach run
    - add tp + fsdp tests