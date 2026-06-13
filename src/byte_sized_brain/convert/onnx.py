"""HuggingFace DistilBERT → ONNX (FP32 export + dynamic INT8 quantization).

The original project only ever kept the INT8 file, so its evaluator — which
needs *both* an FP32 and an INT8 graph — could never run. Here we always export
and keep the FP32 graph first, then derive INT8 from it.

The model is wrapped so the exported graph has a single clean ``logits`` output
rather than a HuggingFace dataclass.
"""

from __future__ import annotations

from pathlib import Path


def export_fp32(model_dir: str | Path, out_path: str | Path, *, seq_len: int = 128, opset: int = 14) -> Path:
    import torch
    from transformers import AutoTokenizer, DistilBertForSequenceClassification

    model = DistilBertForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    class _LogitsOnly(torch.nn.Module):
        def __init__(self, m: torch.nn.Module) -> None:
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):  # noqa: D401
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

    enc = tokenizer(
        "This is a sample input for ONNX export.",
        return_tensors="pt",
        max_length=seq_len,
        truncation=True,
        padding="max_length",
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        _LogitsOnly(model),
        (enc["input_ids"], enc["attention_mask"]),
        str(out),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=opset,
    )
    return out


def quantize_int8(fp32_path: str | Path, int8_path: str | Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out = Path(int8_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(fp32_path), str(out), weight_type=QuantType.QInt8)
    return out
