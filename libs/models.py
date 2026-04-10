from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import yaml
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoModelForImageTextToText,  # NEW (preferred)
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    LlavaForConditionalGeneration,
    AutoTokenizer
)
from transformers import logging

logging.set_verbosity_error()

# `AutoModelForVision2Seq` is deprecated and will be removed in v5.0. Please use `AutoModelForImageTextToText`

from libs.utils import _dtype_from_spec, _to_pil, _load_image

import google.genai as genai
from google.genai import types
import anthropic
from openai import OpenAI


# ----------------------------
# Specs + YAML loader
# ----------------------------

@dataclass(frozen=True)
class ModelSpec:
    provider: str  # "hf" or "openai"
    model_id: str  # HF repo id or OpenAI model name
    trust_remote_code: bool = False
    dtype: str = "auto"  # "auto" | "float16" | "bfloat16" | "float32"
    default_max_new_tokens: int = 1024


def load_model_zoo(yaml_path: Union[str, Path]) -> Dict[str, ModelSpec]:
    """
    Reads:
      models:
        alias:
          provider: hf|openai
          model_id: ...
          trust_remote_code: bool
          dtype: auto|float16|bfloat16|float32
          default_max_new_tokens: int
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML settings not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    models = cfg.get("models", {})
    if not isinstance(models, dict) or not models:
        raise ValueError(f"Invalid YAML: expected non-empty top-level 'models:' mapping in {yaml_path}")

    zoo: Dict[str, ModelSpec] = {}
    for alias, spec in models.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Invalid model spec for '{alias}': expected mapping, got {type(spec)}")

        provider = spec.get("provider")
        model_id = spec.get("model_id")
        if provider not in ("hf", "openai", "gemini", "anthropic"):
            raise ValueError(f"Model '{alias}': provider must be 'hf' or 'openai' (got {provider})")
        if not model_id or not isinstance(model_id, str):
            raise ValueError(f"Model '{alias}': model_id must be a non-empty string")

        zoo[alias] = ModelSpec(
            provider=provider,
            model_id=model_id,
            trust_remote_code=bool(spec.get("trust_remote_code", False)),
            dtype=str(spec.get("dtype", "auto")),
            default_max_new_tokens=int(spec.get("default_max_new_tokens", 256)),
        )

    return zoo


# Load once (or load in main and pass in)
MODEL_ZOO: Dict[str, ModelSpec] = load_model_zoo("evaluation_settings.yaml")


def list_models() -> List[str]:
    return sorted(MODEL_ZOO.keys())


with open("system_prompt.txt", "r", encoding="utf-8") as f:
    NCAP_SYSTEM_PROMPT = f.read().strip()


# ----------------------------
# Backends
# ----------------------------

class BaseVLM:
    def generate(
            self,
            prompt: Union[str, Dict[str, Any], List[Dict[str, Any]]],
            images: Optional[Sequence[Union[str, Image.Image, bytes]]] = None,
            **gen_kwargs: Any,
    ) -> str:
        raise NotImplementedError


class HuggingFaceVLM(BaseVLM):
    def __init__(
            self,
            model_id: str,
            trust_remote_code: bool = False,
            dtype: str = "auto",
            device: Optional[str] = None,
    ):
        self.model_id = model_id
        self.trust_remote_code = trust_remote_code

        torch_dtype = _dtype_from_spec(dtype)
        self.device = device

        self.tokenizer = None

        if self._is_phi(model_id):
            self.processor = AutoProcessor.from_pretrained(model_id,
                                                           trust_remote_code=True,
                                                           num_crops=4
                                                           )
        elif self._is_paligemma(model_id):
            self.processor = PaliGemmaProcessor.from_pretrained(model_id)
        else:
            self.processor = AutoProcessor.from_pretrained(model_id,
                                                           trust_remote_code=trust_remote_code,
                                                           )

        self.access_token = os.getenv("HF_TOKEN")

        load_kwargs = dict(
            device_map="auto",
            trust_remote_code=trust_remote_code,
            token=self.access_token,
            # attn_implementation="eager",
        )

        # LLaVA needs the VLM class (not AutoModelForCausalLM)
        if self._is_llava(model_id):
            self.model = LlavaForConditionalGeneration.from_pretrained(model_id, **load_kwargs).eval().to(self.device)
        elif self._is_phi(model_id):
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs).eval().to(self.device)
        elif self._is_idefics(model_id):
            self.model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(self.device)
        elif self._is_internvl2(model_id) or self._is_minicpm(model_id):
            self.model = AutoModel.from_pretrained(model_id, **load_kwargs).eval().to(self.device)
        elif self._is_paligemma(model_id):
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, **load_kwargs).eval().to(
                self.device)
        else:
            # Preferred for modern VLMs (image+text -> text)
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs).eval().to(self.device)
            except Exception:
                # Backwards compatibility for older model types
                try:
                    self.model = AutoModelForVision2Seq.from_pretrained(model_id, **load_kwargs).eval().to(self.device)
                except Exception:
                    # Last resort: text-only models
                    self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs).eval().to(self.device)

        if device is not None:
            self.model.to(device)

        self.model.eval()

    @staticmethod
    def _has_chat_template(proc) -> bool:
        """
        True if processor/tokenizer can apply a chat template.
        Works across many processor/tokenizer combos.
        """
        if hasattr(proc, "apply_chat_template"):
            return True
        tok = getattr(proc, "tokenizer", None)
        return bool(getattr(tok, "chat_template", None))

    @staticmethod
    def _apply_chat_template(proc, messages):
        """
        Use processor.apply_chat_template if present, else tokenizer.apply_chat_template.
        """
        if hasattr(proc, "apply_chat_template"):
            return proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tok = getattr(proc, "tokenizer", None)
        if tok is not None and hasattr(tok, "apply_chat_template"):
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        raise AttributeError("No chat template available on processor/tokenizer.")

    @staticmethod
    def _extract_user_text(messages: List[Dict[str, Any]]) -> str:
        for m in reversed(messages):
            if not isinstance(m, dict):
                continue
            if m.get("role") != "user":
                continue
            content = m.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for item in reversed(content):
                    if isinstance(item, dict) and item.get("type") == "text":
                        t = item.get("text")
                        if isinstance(t, str):
                            return t
        return ""

    @staticmethod
    def _attach_images(messages: List[Dict[str, Any]], pil_images: Optional[List[Image.Image]]) -> List[Dict[str, Any]]:
        if not pil_images:
            return messages

        img_idx = 0
        out: List[Dict[str, Any]] = []
        for m in messages:
            if not isinstance(m, dict):
                continue

            if m.get("role") != "user":
                out.append(m)
                continue

            content = m.get("content")
            if not isinstance(content, list):
                out.append(m)
                continue

            new_content = []
            for item in content:
                if (
                        isinstance(item, dict)
                        and item.get("type") == "image"
                        and "image" not in item
                        and img_idx < len(pil_images)
                ):
                    new_item = dict(item)
                    new_item["image"] = pil_images[img_idx]
                    img_idx += 1
                    new_content.append(new_item)
                else:
                    new_content.append(item)

            new_m = dict(m)
            new_m["content"] = new_content
            out.append(new_m)

        return out

    @staticmethod
    def _is_internvl3(model_id: str) -> bool:
        s = model_id.lower()
        return "internvl3" in s

    @staticmethod
    def _is_molmo(model_id: str) -> bool:
        s = model_id.lower()
        return "molmo" in s

    @staticmethod
    def _is_minicpm(model_id: str) -> bool:
        s = model_id.lower()
        return "minicpm" in s

    @staticmethod
    def _is_paligemma(model_id: str) -> bool:
        s = model_id.lower()
        return "paligemma" in s

    @staticmethod
    def _is_idefics(model_id: str) -> bool:
        s = model_id.lower()
        return "idefics" in s

    @staticmethod
    def _is_internvl2(model_id: str) -> bool:
        s = model_id.lower()
        return "internvl2" in s

    @staticmethod
    def _is_phi(model_id: str) -> bool:
        s = model_id.lower()
        return "phi" in s

    @staticmethod
    def _is_llava(model_id: str) -> bool:
        s = model_id.lower()
        return "llava" in s

    @staticmethod
    def _is_qwen_vl(model_id: str) -> bool:
        s = model_id.lower()
        return ("qwen2.5-vl" in s) or ("qwen3-vl" in s) or ("qwen2_vl" in s) or ("qwen3_vl" in s)

    @torch.inference_mode()
    def generate(
            self,
            prompt: str,
            images: Optional[Sequence[Union[str, Image.Image, bytes]]] = None,
            max_new_tokens: int = 256,
            temperature: float = 0.2,
            top_p: float = 0.9,
            do_sample: bool = True,
            **gen_kwargs: Any,
    ) -> str:
        pil_images = [_to_pil(im) for im in images] if images else None
        messages_in: Optional[List[Dict[str, Any]]] = None

        if isinstance(prompt, str):
            prompt_text = prompt
        elif isinstance(prompt, dict):
            messages_in = [prompt]
            prompt_text = self._extract_user_text(messages_in)
        elif isinstance(prompt, list) and (len(prompt) == 0 or isinstance(prompt[0], dict)):
            messages_in = prompt  # type: ignore[assignment]
            prompt_text = self._extract_user_text(messages_in or [])
        else:
            prompt_text = str(prompt)
        generation_config = dict()
        pixel_values = 0
        num_patches_list = 0

        is_qwen_vl = self._is_qwen_vl(self.model_id)
        if not is_qwen_vl:
            cfg = getattr(self.model, "config", None)
            mt = str(getattr(cfg, "model_type", "")).lower() if cfg is not None else ""
            arch = getattr(cfg, "architectures", None) if cfg is not None else None
            arch_s = " ".join([str(a) for a in arch]) if isinstance(arch, list) else ""
            if "qwen" in mt and "vl" in mt:
                is_qwen_vl = True
            elif "qwen" in arch_s.lower() and "vl" in arch_s.lower():
                is_qwen_vl = True

        # -------- Build inputs --------
        if pil_images and (is_qwen_vl or self._is_llava(self.model_id)):
            messages = [
                {"role": "system", "content": [{"type": "text", "text": NCAP_SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": (
                            [{"type": "image", "image": im} for im in pil_images] +
                            [{"type": "text", "text": prompt_text}]
                    ),
                },
            ]
            if not self._has_chat_template(self.processor):
                # Some LLaVA variants may still work without templates, but most instruct ones need them.
                # Fallback to plain processor if template missing:
                inputs = self.processor(text=prompt_text, images=pil_images, return_tensors="pt")
            else:
                text = self._apply_chat_template(self.processor, messages)
                inputs = self.processor(text=[text], images=pil_images, return_tensors="pt")
        elif self._is_idefics(self.model_id):
            # Create inputs
            n_imgs = len(pil_images) if pil_images else 0
            messages = [
                {"role": "system", "content": NCAP_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": ([{"type": "image"}] * n_imgs) + [{"type": "text", "text": prompt}],
                },
            ]
            text = prompt
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=pil_images, return_tensors="pt")

        elif self._is_internvl2(self.model_id):
            # set the max number of tiles in `max_num`
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

            # Convert each input image to InternVL pixel_values (patch tokens)
            # Assumes your libs.utils._load_image can take PIL or a path/bytes.
            pixel_values_list = [
                _load_image(im, max_num=12).to(torch.bfloat16).to(next(self.model.parameters()).device)
                for im in pil_images
            ]
            num_patches_list = [pv.size(0) for pv in pixel_values_list]
            pixel_values = torch.cat(pixel_values_list, dim=0)

            generation_config = dict(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )

            # Use "separate images" format: Image-1: <image> ... Image-5: <image>
            text = "\n".join([f"Image-{k + 1}: <image>" for k in range(len(pil_images))]) + "\n" + prompt

            # IMPORTANT: we return via model.chat, so set inputs/text accordingly
            inputs = None

        elif self._is_minicpm(self.model_id):
            generation_config = dict(
                max_new_tokens=256,
                do_sample=True
            )

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            messages = [
                {"role": "system", "content": NCAP_SYSTEM_PROMPT},
                {"role": "user", "content": [*pil_images, prompt]},
            ]
            text = prompt
            inputs = messages

        elif self._is_paligemma(self.model_id):
            inputs = self.processor(
                text=prompt,
                images=pil_images,
                return_tensors="pt").to(torch.bfloat16).to(self.model.device)

        elif self._has_chat_template(self.processor):
            # Inject N image placeholders (5 in your case) + question text
            if self._is_internvl3(self.model_id) or self._is_molmo(self.model_id):
                user_content = []

                if pil_images:
                    user_content += [
                        {"type": "image", "url": im}  # im must be a URL string
                        for im in pil_images
                    ]

                user_content += [
                    {"type": "text", "text": prompt}
                ]

                messages = [
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ]
            else:
                user_content = []
                if pil_images:
                    user_content += [dict(type="image", image=im) for im in pil_images]
                user_content += [dict(type="text", text=prompt)]

                messages = [
                    {"role": "system", "content": NCAP_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ]

            text = prompt
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )

        else:
            # No chat template available (Phi3VProcessor commonly lands here)
            # Use plain text + images.
            text = '<|system|>' + NCAP_SYSTEM_PROMPT + '<|user|>' + '<image>\n' + prompt
            inputs = self.processor(
                text=text,
                images=pil_images,
                return_tensors="pt",
            )

        # Always align input tensors with model device
        if self.device is not None:
            target_device = torch.device(self.device)
        else:
            target_device = next(self.model.parameters()).device

        if self._is_internvl2(self.model_id) or self._is_minicpm(self.model_id):
            print("No need to change the inputs")
        elif self._is_idefics(self.model_id) or self._is_molmo(self.model_id):
            inputs = {k: v.to(target_device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(target_device) if torch.is_tensor(v) else v for k, v in inputs.items()}

        if self._is_internvl2(self.model_id):
            output_ids, _ = self.model.chat(
                self.tokenizer,
                pixel_values,
                text,
                generation_config,
                num_patches_list=num_patches_list,
            )

        if self._is_minicpm(self.model_id):
            output_ids = self.model.chat(
                tokenizer=self.tokenizer,
                image=None,
                msgs=text
            )
        else:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                **gen_kwargs,
            )

        if self._is_internvl2(self.model_id) or self._is_minicpm(self.model_id):
            out = output_ids
        elif hasattr(self.processor, "tokenizer"):
            out = self.processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        else:
            out = self.processor.decode(output_ids[0], skip_special_tokens=True)

        if hasattr(self.processor, "apply_chat_template") and isinstance(prompt, str) and prompt in out:
            out = out.split(prompt, 1)[-1].strip()

        out = out.lower().split("assistant")[-1].strip()

        return out.strip()


class GeminiVLM(BaseVLM):
    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        api_key = os.environ.get("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)

    def generate(
            self,
            prompt: str,
            images: Optional[Sequence[Union[str, Image.Image, bytes]]] = None,
            max_output_tokens: int = 256,
            **kwargs: Any,
    ) -> str:

        content: List[Any] = []

        # Add images (Gemini accepts PIL directly)
        if images:
            for im in images:
                content.append(_to_pil(im))

        # Add user question
        content.append(prompt)

        resp = self.client.models.generate_content(
            model=self.model,
            contents=content,
            config=types.GenerateContentConfig(
                system_instruction=NCAP_SYSTEM_PROMPT,
                max_output_tokens=max_output_tokens,
            ),
        )

        return resp.text.strip()


class ClaudeVLM(BaseVLM):
    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key)

    def _image_to_base64(self, img: Union[str, Image.Image, bytes]) -> str:
        pil = _to_pil(img)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def generate(
            self,
            prompt: str,
            images: Optional[Sequence[Union[str, Image.Image, bytes]]] = None,
            max_output_tokens: int = 256,
            **kwargs: Any,
    ) -> str:

        content: List[Dict[str, Any]] = []

        # Add images first
        if images:
            for im in images:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": self._image_to_base64(im),
                    },
                })

        # Add question
        content.append({
            "type": "text",
            "text": prompt,
        })

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_output_tokens,
            system=NCAP_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            **kwargs,
        )

        return (resp.content[0].text or "").strip()


class OpenAIVLM(BaseVLM):
    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        api_key = os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

    def _image_to_data_url(self, img: Union[str, Image.Image, bytes]) -> str:
        pil = _to_pil(img)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def generate(
            self,
            prompt: str,
            images: Optional[Sequence[Union[str, Image.Image, bytes]]] = None,
            max_output_tokens: int = 256,
            **kwargs: Any,
    ) -> str:
        content: List[Dict[str, Any]] = []

        # Add images first (for multi-image VLM context)
        if images:
            for im in images:
                content.append({
                    "type": "input_image",
                    "image_url": self._image_to_data_url(im)
                })

        # Then add question text
        content.append({
            "type": "input_text",
            "text": prompt
        })

        resp = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": NCAP_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": content,
                }
            ],
            max_output_tokens=max_output_tokens,
            **kwargs,
        )

        return (getattr(resp, "output_text", None) or "").strip()


# ----------------------------
# Unified facade (now also accepts custom zoo)
# ----------------------------

class VLM(BaseVLM):
    def __init__(
            self,
            model: str,
            *,
            model_zoo: Optional[Dict[str, ModelSpec]] = None,
            hf_device: Optional[str] = None,
            hf_dtype: Optional[str] = None,
            openai_api_key: Optional[str] = None,
    ):
        self.model_zoo = model_zoo or MODEL_ZOO

        spec: Optional[ModelSpec] = None
        if model.startswith("hf:"):
            hf_id = model[len("hf:"):]
            spec = ModelSpec(provider="hf", model_id=hf_id, trust_remote_code=True)
        elif model.startswith("openai:") and model not in self.model_zoo:
            oa_model = model[len("openai:"):]
            spec = ModelSpec(provider="openai", model_id=oa_model)
        elif model.startswith("gemini:") and model not in self.model_zoo:
            oa_model = model[len("gemini:"):]
            spec = ModelSpec(provider="gemini", model_id=oa_model)
        elif model.startswith("anthropic:") and model not in self.model_zoo:
            oa_model = model[len("anthropic:"):]
            spec = ModelSpec(provider="anthropic", model_id=oa_model)
        else:
            spec = self.model_zoo.get(model)

        if spec is None:
            raise ValueError(
                f"Unknown model '{model}'. Available aliases:\n  " + "\n  ".join(sorted(self.model_zoo.keys()))
            )

        self.spec = spec

        if spec.provider == "hf":
            self.backend = HuggingFaceVLM(
                model_id=spec.model_id,
                trust_remote_code=spec.trust_remote_code,
                device="cuda",
            )
        elif spec.provider == "openai":
            self.backend = OpenAIVLM(model=spec.model_id, api_key=openai_api_key)
        elif spec.provider == "gemini":
            self.backend = GeminiVLM(model=spec.model_id, api_key=openai_api_key)
        elif spec.provider == "anthropic":
            self.backend = ClaudeVLM(model=spec.model_id, api_key=openai_api_key)
        else:
            raise ValueError(f"Unsupported provider: {spec.provider}")

    def generate(
            self,
            prompt: str,
            images: Optional[Sequence[Union[str, Image.Image, bytes]]] = None,
            **gen_kwargs: Any,
    ) -> str:
        if self.spec.provider == "hf":
            gen_kwargs.setdefault("max_new_tokens", self.spec.default_max_new_tokens)
        else:
            gen_kwargs.setdefault("max_output_tokens", self.spec.default_max_new_tokens)
        return self.backend.generate(prompt=prompt, images=images, **gen_kwargs)
