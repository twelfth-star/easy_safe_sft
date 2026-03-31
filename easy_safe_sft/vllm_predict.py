from pathlib import Path
from typing import Any

from loguru import logger

from easy_safe_sft.utils import read_jsonl, write_jsonl


def _split_messages(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], str]:
    if not messages:
        raise ValueError("messages 不能为空")

    prompt_messages = messages
    label = ""
    if messages[-1]["role"] == "assistant":
        prompt_messages = messages[:-1]
        label = messages[-1]["content"]

    return prompt_messages, label


def _render_qwen3_nothink_prompt(messages: list[dict[str, str]]) -> str:
    """
    NOTE: Qwen3系列的原生tokenizer的apply_chat_template涉及到一个<think> token的问题，在training和inference时的表现不太一致，比较复杂。
    我们这里参考llama factory的设计，统一成下面这个形式。从而使得training和inference时的格式统一。
    """
    parts: list[str] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def _render_prompt(messages: list[dict[str, str]], template: str, tokenizer: Any) -> str:
    if template in ("qwen3_nothink", "qwen3_5_nothink", "qwen3_5"):
        return _render_qwen3_nothink_prompt(messages)

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    raise ValueError(f"当前纯 vLLM 评测不支持模板: {template}")


def _visible_gpu_count() -> int:
    cuda_visible_devices = Path("/proc/self/environ").read_bytes().decode("utf-8", errors="ignore")
    if "CUDA_VISIBLE_DEVICES=" not in cuda_visible_devices:
        try:
            import torch

            return max(1, int(torch.cuda.device_count()))
        except Exception:
            return 1

    raw = cuda_visible_devices.split("CUDA_VISIBLE_DEVICES=", 1)[1].split("\x00", 1)[0].strip()
    if not raw:
        return 1
    return len([item for item in raw.split(",") if item.strip()])


def _cuda_visible_devices() -> str:
    environ = Path("/proc/self/environ").read_bytes().decode("utf-8", errors="ignore")
    if "CUDA_VISIBLE_DEVICES=" not in environ:
        return ""
    return environ.split("CUDA_VISIBLE_DEVICES=", 1)[1].split("\x00", 1)[0]


class VllmPredictor:
    def __init__(
        self,
        base_config: dict[str, Any],
        adapter_path: str | None = None,
        concurrency: int = 32,
        tp_size: int | None = None,
        gpu_util: float | None = None,
        maxlen: int | None = None,
    ) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        finetuning_type = str(base_config.get("finetuning_type", "lora"))
        base_model_path = str(base_config["model_name_or_path"])
        model_path = base_model_path
        if finetuning_type != "lora" and adapter_path is not None:
            model_path = adapter_path

        tokenizer_path = base_model_path if finetuning_type == "lora" else model_path
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=bool(base_config.get("trust_remote_code", False)),
        )
        self._template = str(base_config["template"])
        self._concurrency = max(1, concurrency)
        self._sampling_params = SamplingParams(
            temperature=float(base_config.get("temperature", 1.0)) if bool(base_config.get("do_sample", False)) else 0.0,
            top_p=float(base_config.get("top_p", 1.0)),
            top_k=int(base_config.get("top_k", -1)),
            repetition_penalty=float(base_config.get("repetition_penalty", 1.0)),
            max_tokens=int(base_config.get("max_new_tokens", 16)),
            stop_token_ids=[self._tokenizer.convert_tokens_to_ids("<|im_end|>")],
        )

        resolved_tp_size = _visible_gpu_count() if tp_size is None else tp_size
        llm_kwargs: dict[str, Any] = {
            "model": model_path,
            "tokenizer": tokenizer_path,
            "trust_remote_code": bool(base_config.get("trust_remote_code", False)),
            "tensor_parallel_size": resolved_tp_size,
            "dtype": str(base_config.get("infer_dtype", "auto")),
            "gpu_memory_utilization": float(gpu_util if gpu_util is not None else base_config.get("vllm_gpu_util", 0.9)),
            "max_model_len": int(maxlen if maxlen is not None else base_config.get("vllm_maxlen", 4096)),
        }

        self._lora_request = None
        if finetuning_type == "lora" and adapter_path is not None:
            lora_rank = int(base_config.get("lora_rank", 32))
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = lora_rank
            self._lora_request = LoRARequest(
                lora_name=Path(adapter_path).name,
                lora_int_id=1,
                lora_path=adapter_path,
                base_model_name=base_model_path,
            )

        self._llm = LLM(**llm_kwargs)
        logger.info(
            "vLLM 评测器已启动: model_path={}, adapter_path={}, template={}, concurrency={}, tp_size={}, gpu_util={}, maxlen={}, CUDA_VISIBLE_DEVICES={}",
            model_path,
            adapter_path,
            self._template,
            self._concurrency,
            resolved_tp_size,
            llm_kwargs["gpu_memory_utilization"],
            llm_kwargs["max_model_len"],
            _cuda_visible_devices(),
        )

    def predict_dataset(self, dataset_path: str, output_path: str, progress_desc: str) -> None:
        from tqdm.auto import tqdm

        rows = read_jsonl(dataset_path)
        prompts: list[str] = []
        labels: list[str] = []
        for row in rows:
            prompt_messages, label = _split_messages(row["messages"])
            prompts.append(_render_prompt(prompt_messages, self._template, self._tokenizer))
            labels.append(label)

        predictions: list[dict[str, str]] = []
        with tqdm(total=len(prompts), desc=progress_desc, unit="example") as progress:
            for start in range(0, len(prompts), self._concurrency):
                prompt_batch = prompts[start : start + self._concurrency]
                label_batch = labels[start : start + self._concurrency]
                outputs = self._llm.generate(
                    prompt_batch,
                    self._sampling_params,
                    use_tqdm=False,
                    lora_request=self._lora_request,
                )
                for prompt, label, output in zip(prompt_batch, label_batch, outputs):
                    prediction = output.outputs[0].text if output.outputs else ""
                    predictions.append(
                        {
                            "prompt": prompt,
                            "predict": prediction,
                            "label": label,
                        }
                    )
                progress.update(len(prompt_batch))

        write_jsonl(output_path, predictions)
        logger.info("vLLM 预测完成: dataset={}, output={}", dataset_path, output_path)

    def close(self) -> None:
        del self._llm
