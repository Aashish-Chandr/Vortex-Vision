"""
Qwen2.5-VL / Qwen3-VL integration for multimodal video understanding
and natural language querying over video content.
"""
import base64
import logging
import numpy as np
from typing import List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


def _encode_image(frame: np.ndarray) -> str:
    """Encode a BGR numpy frame to base64 JPEG string."""
    import cv2
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


class QwenVLClient:
    """
    Client for Qwen2.5-VL / Qwen3-VL.
    Supports local inference via transformers or remote via OpenAI-compatible API.
    """

    def __init__(
        self,
        mode: str = "local",           # "local" | "api"
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        api_base: str = "http://localhost:8080/v1",
        api_key: str = "EMPTY",
        device: str = "cuda",
        max_new_tokens: int = 512,
    ):
        self.mode = mode
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        if mode == "local":
            self._load_local(model_name, device)
        else:
            self._init_api(api_base, api_key)

    def _load_local(self, model_name: str, device: str):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        logger.info("Loading %s locally...", model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device,
        )
        logger.info("Model loaded.")

    def _init_api(self, api_base: str, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(base_url=api_base, api_key=api_key)

    def query_frames(
        self,
        frames: List[np.ndarray],
        question: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Ask a natural language question about a list of video frames.
        Returns the model's text response.
        """
        if self.mode == "local":
            return self._local_query(frames, question, system_prompt)
        return self._api_query(frames, question, system_prompt)

    def _build_messages(self, frames: List[np.ndarray], question: str, system_prompt: Optional[str]):
        content = []
        for frame in frames[:8]:  # cap at 8 frames per query
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(frame)}"},
            })
        content.append({"type": "text", "text": question})

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        return messages

    def _local_query(self, frames: List[np.ndarray], question: str, system_prompt: Optional[str]) -> str:
        import torch
        messages = self._build_messages(frames, question, system_prompt)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0]

    def _api_query(self, frames: List[np.ndarray], question: str, system_prompt: Optional[str]) -> str:
        messages = self._build_messages(frames, question, system_prompt)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_new_tokens,
        )
        return response.choices[0].message.content
