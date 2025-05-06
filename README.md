Byte-Sized Brain:

A collection of four ML pipelines—FFN on MNIST, CNN on CIFAR-10, RNN on IMDB, and DistilBERT on IMDB—demonstrating training, TFLite/ONNX conversion, and edge-device evaluation with metrics.

Pipelines:
1. FNN on MNIST
  Train a feed-forward network on MNIST
  Convert to FP32 & INT8 TFLite
  Evaluate accuracy, latency, RAM Δ, peak memory

2. CNN on CIFAR-10
  Train a simple CNN (MobileNetV2 head) on CIFAR-10
  Convert to FP32 SavedModel & INT8 TFLite
  Evaluate accuracy, latency, RAM Δ, model-size reduction

3. RNN on IMDB
  Train an LSTM classifier on IMDB reviews
  Convert to dynamic-range-quantized TFLite (Flex)
  Evaluate accuracy, latency, RAM Δ, model-size reduction

4. DistilBERT on IMDB
  Fine-tune DistilBERT for sentiment on IMDB
  Export to ONNX FP32 & INT8
  Evaluate accuracy, latency, RAM Δ, peak memory, model-size reduction

Running:
1. Train
Example: CNN pipeline
  python3 train_cnn_cifar.py

2. Convert
Example: CNN FP32 → INT8
  python3 convert_cnn_to_tflite.py

3. Evaluate
Example: CNN TFLite evaluation
  python3 eval_cnn_tflite.py

Each script prints a summary table of metrics.

Requirements:
Install all dependencies with:

  pip install -r requirements.txt

The requirements.txt includes:
tensorflow>=2.13.0           # training & TFLite conversion
torch>=2.0.0                 # ONNX export for DistilBERT
onnxruntime>=1.15.0          # ONNX inference on CPU
transformers>=4.45.0         # 🤗 Transformers (DistilBERT)
datasets>=2.16.0             # HF Datasets (IMDB, etc.)
numpy>=1.24.0
psutil>=5.9.0                # RAM measurements
pyrapl>=0.4.0                # (optional) energy profiling
accelerate>=0.20.0           # optional 🤗 Trainer support

That’s it—clone the repo, install requirements, then run each pipeline’s train/convert/eval scripts as needed.
