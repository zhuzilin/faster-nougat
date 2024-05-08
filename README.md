## faster-nougat

Implementation of nougat that focuses on processing one file at a time, currently implements with MLX.

### benchmark

```bash
# download test pdf
wget https://arxiv.org/pdf/1706.03762 -O 1706.03762v7.pdf

# baseline
python baseline.py

# mlx baseline
python baseline_with_mlx.py
```
