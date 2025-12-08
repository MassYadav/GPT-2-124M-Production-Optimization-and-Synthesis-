# ✨ Production-Ready GPT-2 (124M) Synthesis and Optimization

## Summary
A comprehensive reproduction and performance-tuning project for the 124 million parameter GPT-2 model. This project emphasizes MLOps efficiency and low-latency training, resulting in **11x faster throughput** and a model that **surpassed the original GPT-2 checkpoint on HellaSwag accuracy**.

This work demonstrates expertise in deep learning performance engineering and maximizing GPU utilization for large-scale GenAI systems.

## 🚀 Key Performance Engineering Achievements

This project successfully integrates high-impact training optimizations required for production-scale LLM development:

* **11x Training Speed-Up:** Achieved substantial increases in training efficiency by implementing:
    * **Flash Attention Fused Kernels:** Utilized memory-efficient fused attention to avoid materializing large attention matrices, significantly reducing VRAM footprint and accelerating computations.
    * **Mixed Precision Training (BF16/FP16):** Leveraged PyTorch's Automatic Mixed Precision (`torch.cuda.amp`) to reduce memory bandwidth requirements without compromising model stability.
    * **Gradient Accumulation:** Effectively scaled the batch size for stable training dynamics while working within GPU memory constraints.

* **GPT-2 Architecture from Scratch:** Built the full GPT-2 architecture in PyTorch, ensuring faithful reproduction of:
    * Multi-Head Attention (MHA)
    * Layer Normalization (LayerNorm)
    * Positional Encoding

* **Validation & Benchmarking:**
    * The model was successfully trained using the OpenWebText corpus.
    * Performance was validated via the **HellaSwag benchmark**, confirming the model's ability to exceed the original GPT-2 (124M) checkpoint accuracy.

## ⚙️ Project Environment
* **Framework:** PyTorch, Python
* **Libraries:** Flash Attention, CUDA AMP
* **MLOps Focus:** Demonstrates expertise in performance bottlenecks, resource allocation, and model synthesis.

## Usage (Optimized Training)

The following command executes the training script with all key optimizations enabled:

```bash
python train.py --model_size 124M --mixed_precision bf16 --use_flash_attention --gradient_accumulation_steps 8



## CLONE REPO
git clone https://github.com/rowanz/hellaswag.git
