# -*- coding: utf-8 -*-

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
import locale
import torchaudio.transforms as T
import os
from snac import SNAC
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

locale.getpreferredencoding = lambda: "UTF-8"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/orpheus-3b-0.1-ft",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

dataset = load_dataset("MrDragonFox/Elise", split="train")
ds_sample_rate = dataset[0]["audio"]["sampling_rate"]

snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
snac_model = snac_model.to("cuda")

def tokenise_audio(waveform):
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    waveform = waveform.to(dtype=torch.float32)
    resample_transform = T.Resample(orig_freq=ds_sample_rate, new_freq=24000)
    waveform = resample_transform(waveform)
    waveform = waveform.unsqueeze(0).to("cuda")

    with torch.inference_mode():
        codes = snac_model.encode(waveform)

    all_codes = []
    for i in range(codes[0].shape[1]):
        all_codes.append(codes[0][0][i].item()+128266)
        all_codes.append(codes[1][0][2*i].item()+128266+4096)
        all_codes.append(codes[2][0][4*i].item()+128266+(2*4096))
        all_codes.append(codes[2][0][(4*i)+1].item()+128266+(3*4096))
        all_codes.append(codes[1][0][(2*i)+1].item()+128266+(4*4096))
        all_codes.append(codes[2][0][(4*i)+2].item()+128266+(5*4096))
        all_codes.append(codes[2][0][(4*i)+3].item()+128266+(6*4096))

    return all_codes

def add_codes(example):
    codes_list = None

    try:
        answer_audio = example.get("audio")
        if answer_audio and "array" in answer_audio:
            audio_array = answer_audio["array"]
            codes_list = tokenise_audio(audio_array)
    except Exception as e:
        print(f"Skipping row due to error: {e}")
    example["codes_list"] = codes_list
    return example

dataset = dataset.map(add_codes, remove_columns=["audio"])

tokeniser_length = 128256
start_of_text = 128000
end_of_text = 128009
start_of_speech = tokeniser_length + 1
end_of_speech = tokeniser_length + 2
start_of_human = tokeniser_length + 3
end_of_human = tokeniser_length + 4
start_of_ai = tokeniser_length + 5
end_of_ai = tokeniser_length + 6
pad_token = tokeniser_length + 7
audio_tokens_start = tokeniser_length + 10

dataset = dataset.filter(lambda x: x["codes_list"] is not None)
dataset = dataset.filter(lambda x: len(x["codes_list"]) > 0)

def remove_duplicate_frames(example):
    vals = example["codes_list"]
    if len(vals) % 7 != 0:
        raise ValueError("Input list length must be divisible by 7")

    result = vals[:7]
    removed_frames = 0

    for i in range(7, len(vals), 7):
        current_first = vals[i]
        previous_first = result[-7]

        if current_first != previous_first:
            result.extend(vals[i:i+7])
        else:
            removed_frames += 1

    example["codes_list"] = result
    return example

dataset = dataset.map(remove_duplicate_frames)

def create_input_ids(example):
    text_prompt = f"{example['source']}: {example['text']}" if "source" in example else example["text"]
    text_ids = tokenizer.encode(text_prompt, add_special_tokens=True)
    text_ids.append(end_of_text)

    example["text_tokens"] = text_ids
    input_ids = (
        [start_of_human]
        + example["text_tokens"]
        + [end_of_human]
        + [start_of_ai]
        + [start_of_speech]
        + example["codes_list"]
        + [end_of_speech]
        + [end_of_ai]
    )
    example["input_ids"] = input_ids
    example["labels"] = input_ids
    example["attention_mask"] = [1] * len(input_ids)
    return example

dataset = dataset.map(create_input_ids, remove_columns=["text", "codes_list"])
columns_to_keep = ["input_ids", "labels", "attention_mask"]
columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
dataset = dataset.remove_columns(columns_to_remove)

trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

prompts = [
    "Hey there my name is Elise, <giggles> and I'm a speech generation model that can sound like a person.",
]

chosen_voice = None

FastLanguageModel.for_inference(model)
snac_model.to("cpu")

prompts_ = [(f"{chosen_voice}: " + p) if chosen_voice else p for p in prompts]

all_input_ids = []
for prompt in prompts_:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    all_input_ids.append(input_ids)

start_token = torch.tensor([[128259]], dtype=torch.int64)
end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)

all_modified_input_ids = []
for input_ids in all_input_ids:
    modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)
    all_modified_input_ids.append(modified_input_ids)

all_padded_tensors = []
all_attention_masks = []
max_length = max([modified_input_ids.shape[1] for modified_input_ids in all_modified_input_ids])
for modified_input_ids in all_modified_input_ids:
    padding = max_length - modified_input_ids.shape[1]
    padded_tensor = torch.cat([torch.full((1, padding), 128263, dtype=torch.int64), modified_input_ids], dim=1)
    attention_mask = torch.cat([torch.zeros((1, padding), dtype=torch.int64), torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64)], dim=1)
    all_padded_tensors.append(padded_tensor)
    all_attention_masks.append(attention_mask)

all_padded_tensors = torch.cat(all_padded_tensors, dim=0)
all_attention_masks = torch.cat(all_attention_masks, dim=0)

input_ids = all_padded_tensors.to("cuda")
attention_mask = all_attention_masks.to("cuda")
generated_ids = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=1200,
    do_sample=True,
    temperature=0.6,
    top_p=0.95,
    repetition_penalty=1.1,
    num_return_sequences=1,
    eos_token_id=128258,
    use_cache=True
)

token_to_find = 128257
token_to_remove = 128258

token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

if len(token_indices[1]) > 0:
    last_occurrence_idx = token_indices[1][-1].item()
    cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
else:
    cropped_tensor = generated_ids

mask = cropped_tensor != token_to_remove

processed_rows = []
for row in cropped_tensor:
    masked_row = row[row != token_to_remove]
    processed_rows.append(masked_row)

code_lists = []
for row in processed_rows:
    row_length = row.size(0)
    new_length = (row_length // 7) * 7
    trimmed_row = row[:new_length]
    trimmed_row = [t - 128266 for t in trimmed_row]
    code_lists.append(trimmed_row)

def redistribute_codes(code_list):
    layer_1 = []
    layer_2 = []
    layer_3 = []
    for i in range((len(code_list)+1)//7):
        layer_1.append(code_list[7*i])
        layer_2.append(code_list[7*i+1]-4096)
        layer_3.append(code_list[7*i+2]-(2*4096))
        layer_3.append(code_list[7*i+3]-(3*4096))
        layer_2.append(code_list[7*i+4]-(4*4096))
        layer_3.append(code_list[7*i+5]-(5*4096))
        layer_3.append(code_list[7*i+6]-(6*4096))
    codes = [torch.tensor(layer_1).unsqueeze(0),
             torch.tensor(layer_2).unsqueeze(0),
             torch.tensor(layer_3).unsqueeze(0)]

    audio_hat = snac_model.decode(codes)
    return audio_hat

my_samples = []
for code_list in code_lists:
    samples = redistribute_codes(code_list)
    my_samples.append(samples)

if len(prompts) != len(my_samples):
    raise Exception("Number of prompts and samples do not match")
else:
    for i in range(len(my_samples)):
        print("\n=== Prompt:", prompts[i], "===")
        samples = my_samples[i]
        audio_np = samples.detach().squeeze().to("cpu").numpy()
        print(f"[Saved audio tensor of shape {audio_np.shape}]")

del my_samples, samples

print("\nSaving LoRA model into ./lora_model ...")
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
