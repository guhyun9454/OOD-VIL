
# Environment
- Python 3.8.x
- PyTorch 1.12.1
- Torchvision 0.13.1
- NVIDIA GeForce RTX 3090
- CUDA 11.3


# Getting Started
## Environment
```bash
git clone git@github.com/KHU-AGI/VIL.git
cd VIL
conda create -n OODVIL python==3.8
conda activate OODVIL
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

## Run ICON on VIL with iDigits dataset
```bash
python main.py --dataset iDigits --num_tasks 20 --seed 42 --batch-size 24 --IL_mode vil --model vit_base_patch16_224_ICON --method ICON --IC --thre 0.0 --beta 0.01 --use_cast_loss --k 2 --d_threshold  --develop --verbose
```

## Run FT on VIL with iDigits dataset
```bash
python main.py --dataset iDigits --num_tasks 20 --IL_mode vil --seed 42 --batch-size 24 --method FT
```

## Run FT on CIL with iDigits dataset
```bash
python main.py --dataset iDigits --num_tasks 5 --IL_mode cil --seed 42 --batch-size 24 --method FT
```
## Run FT on DIL with iDigits dataset
```bash
python main.py --dataset iDigits --num_tasks 4 --IL_mode dil --seed 42 --batch-size 24 --method FT
```