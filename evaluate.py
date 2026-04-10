from __future__ import annotations

import datetime
import json
import random
from tqdm import tqdm
from dataclasses import replace
from PIL import Image
import io

from libs.models import list_models, load_model_zoo, VLM
from libs.datasets import DatasetWrapper
from libs.calculate_accuracy import calculate_accuracy
from libs.peft_merge import ensure_merged_if_lora_adapter
from libs.utils import _evaluate_via_judge


# ----------------------------
# Example
# ----------------------------
if __name__ == "__main__":
    debug = False
    multi_choice_prompting = False
    print("Available model aliases:", list_models())

    zoo = load_model_zoo("evaluation_settings.yaml")

    results = {}

    dataset = DatasetWrapper(
        dataset_name="ibarcelo/Automingo_dataset",
        pad_missing_images=True,
        only_split="validation",   # key speedup: no train, both if unused
        map_batch_size=64,
        map_writer_batch_size=64,
    )

    _, val_ds = dataset.get_datasets()
    subset = val_ds.select(range(1055))

    for model_name in list(zoo.keys()):
        print(f"\n--- Running model: {model_name} ---")
        if multi_choice_prompting:
            GT_indexes = []
            answers = []

        provider = zoo[model_name].provider

        # If this HF model_id is actually a LoRA adapter dir, merge it once and replace model_id with merged path.
        if provider == "hf":
            merged_id = ensure_merged_if_lora_adapter(
                zoo[model_name].model_id,
                hf_dtype="bfloat16",
            )
            if merged_id != zoo[model_name].model_id:
                zoo[model_name] = replace(zoo[model_name], model_id=merged_id)

        # Prepare results container for this model (store ALL samples)
        results[model_name] = {
            "provider": provider,
            "samples": [],
        }

        # Load model
        if provider == "hf":
            vlm = VLM(
                model=model_name,
                model_zoo=zoo,
                hf_dtype="bfloat16"
            )
        elif provider in ["openai", "gemini", "anthropic"]:
            vlm = VLM(
                model=model_name,
                model_zoo=zoo
            )
        else:
            vlm = None

        for sample_idx, sample in tqdm(enumerate(subset)):
            raw_images = sample["images"]
            images = []
            for im in raw_images:
                if isinstance(im, dict):
                    if "image" in im:
                        images.append(im["image"])
                    elif "bytes" in im:
                        images.append(Image.open(io.BytesIO(im["bytes"])).convert("RGB"))
                    elif "path" in im:
                        images.append(Image.open(im["path"]).convert("RGB"))
                    else:
                        raise ValueError(f"Unknown image dict format: {im.keys()}")
                else:
                    images.append(im)
            prompt = sample["question"]
            ground_truth = sample["answer"]
            completion = sample["answers_reasoning"]

            # IMPORTANT: don't mutate dataset sample in-place
            multi_choice = list(sample["multi_choice_answers"])
            multi_choice.append(completion)

            random.shuffle(multi_choice)
            GT_index = multi_choice.index(completion) + 1
            multiple_choice_prompt = [prompt + "\n\nOptions:\n" + "\n".join(
                [f"{i+1}. {opt}" for i, opt in enumerate(multi_choice)]
            ) + "\n\nAnswer (just the option number):"][0]

            if multi_choice_prompting:
                prompt = multiple_choice_prompt

            # HF models (like Qwen3-VL) often expect chat-style messages, not a raw string
            prompt_for_model = prompt

            try:
                if vlm is None:
                    answer = "Unsupported provider"
                elif provider == "hf":
                    answer = vlm.generate(
                        prompt_for_model,
                        images=images,
                        max_new_tokens=200
                    )
                elif provider in ["openai", "gemini", "anthropic"]:
                    answer = vlm.generate(
                        prompt_for_model,
                        images=images,
                        max_output_tokens=200
                    )
                else:
                    answer = "Unsupported provider"
    
                if debug:
                    print("Answer:", answer)
    
                score = 0.0
                if not multi_choice_prompting:
                    true_false, score = _evaluate_via_judge(prompt, completion, answer)
    
                results[model_name]["samples"].append({
                    "sample_idx": sample_idx,
                    "prompt": prompt,  # keep string prompt in JSON
                    "answer": answer,
                    "gt_index": GT_index if multi_choice_prompting else None,
                    "lingo_judge": score,
                })
    
                if multi_choice_prompting:
                    GT_indexes.append(GT_index)
                    answers.append(answer)

            except Exception as e:
                print(f"Model {model_name} failed:", str(e))

                results[model_name]["samples"].append({
                    "sample_idx": sample_idx,
                    "prompt": prompt,
                    "error": str(e),
                    "gt_index": GT_index if multi_choice_prompting else None,
                    "lingo_judge": 0.0,
                })

        if multi_choice_prompting:
            final_accuracy, not_processed = calculate_accuracy(GT_indexes, answers, model_name)
            print(f"Final accuracy for {model_name}: {100 * final_accuracy:.2f} %. "
                  f"Not processed: {not_processed} out of {len(GT_indexes)}")

            results[model_name]["final_accuracy"] = final_accuracy
            results[model_name]["not_processed"] = not_processed
            results[model_name]["num_samples"] = len(GT_indexes)

        # cleaning at the of per-model evaluation
        if provider == "hf" and vlm is not None:
            import gc, torch

            try:
                # drop references
                backend = getattr(vlm, "backend", None)
                model = getattr(backend, "model", None) if backend else None
                processor = getattr(backend, "processor", None) if backend else None

                del model
                del processor
                del backend
                del vlm
            except Exception:
                del vlm

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if multi_choice_prompting:
        output_file = f"vlm_results_{timestamp}_mqc.json"
    else:
        output_file = f"vlm_results_{timestamp}_lingo_judge.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nResults saved to {output_file}")