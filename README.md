# Basic MNIST Example

Checking the GPU status
```
python -c "import torch; print('CUDA Engine:', torch.version.cuda); print('GPU Available:', torch.cuda.is_available())"

```

Running the script
```bash
pip install -r requirements.txt
python main.py
```


If main.py doesnt run, check which .venv is being used
```
python -c "import sys; print(sys.executable)"
& "deep-learning-mnist/.venv/Scripts/Activate.ps1"
```