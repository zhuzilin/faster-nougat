## faster-nougat

Implementation of nougat that focuses on processing pdf locally.

I hope this could be a helpful component for a good open source RAG system.

### Installation

```bash
git clone https://github.com/zhuzilin/faster-nougat
cd faster-nougat
pip install .
```

You can then try the example with `simple_arxiv_reader.py` (using [deepseek](https://www.deepseek.com/en) api by default). For example, we could let the llm list the contribution of _Attention Is All You Need_ with its 1,2,10 page of the origin paper.

```bash
python simple_arxiv_reader.py \
    --arxiv_url https://arxiv.org/pdf/1706.03762 \
    --pages 1 2 10 \
    --llm_key $YOUR_LLM_KEY \
    --question "please list the main contribution of the paper."
```

### benchmark

The current benchmark is parsing the second page of the great _Attention Is All You Need_ with [nougat-small](https://huggingface.co/facebook/nougat-small).

On M1 pro, the result is:

|          | huggingface | faster nougat |
| -------- | ----------- | ------------- |
| time/sec | 21.7        | 4.5           |

To reproduce, run:

```bash
# download test pdf
wget https://arxiv.org/pdf/1706.03762 -O 1706.03762v7.pdf

# huggingface impl from:
# https://huggingface.co/docs/transformers/main/en/model_doc/nougat
python benchmark/benchmark_hf.py

# faster nougat impl
python benchmark/benchmark_faster_nougat.py
```

### Rationale

There is no magic here :p, I reimplement the decoder part of nougat in [MLX](https://github.com/ml-explore/mlx), which is much faster than pytorch on apple silicons.

### TODOs

- [ ] Implement encoder in MLX (may not be necessary, as encoder takes little time).
- [ ] Explore the possibility of implement this in [llama.cpp](https://github.com/ggerganov/llama.cpp) or other backends.
