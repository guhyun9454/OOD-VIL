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
python main.py --dataset iDigits --num_tasks 20 --IL_mode vil --method ICON --seed 42 --batch-size 24 --IC  --CAST --d_threshold
```

## Run FT on VIL with iDigits dataset
```bash
python main.py --dataset iDigits --num_tasks 20 --IL_mode vil --method FT --seed 42 --batch-size 24 --lr 0.01
```

## Run FT on CIL with iDigits dataset
```bash
python main.py --dataset iDigits --num_tasks 5 --IL_mode cil --method FT --seed 42 --batch-size 24 
```
## Run FT on DIL with iDigits dataset
```bash
python main.py --dataset iDigits --num_tasks 4 --IL_mode dil --method FT --seed 42 --batch-size 24 
```
---
This codebase is from VIL (Park, MY., Lee, JH., Park, GM. ECCV2024) https://github.com/KU-VGI/VIL

