import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

MODEL_ID = "meta-llama/Llama-3.2-1B"
TRAIN_SAMPLES = 3000
EVAL_SAMPLES = 1000
MAX_SEQ_LEN = 512
OUTPUT_DIR = "./gsm8k_lora_output"

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none"
)

def load_gsm8k():
    dataset = load_dataset("openai/gsm8k", "main")
    train_ds = dataset["train"].shuffle(seed=42).select(range(TRAIN_SAMPLES))
    eval_ds = dataset["test"].shuffle(seed=42).select(range(min(EVAL_SAMPLES, len(dataset["test"]))))
    return train_ds, eval_ds

def format_sample(example):
    return {
        "text": f"Question: {example['question']}\n\nStep-by-step solution:\n{example['answer']}"
    }

def build_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    return model

def train(train_ds, eval_ds, model, tokenizer):
    model = get_peft_model(model, lora_config)

    train_ds = train_ds.map(format_sample)
    eval_ds = eval_ds.map(format_sample)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        args=training_args,
    )

    trainer.train()
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    return trainer

def extract_number(text):
    match = re.search(r'####\s*([\d,]+)', text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    return numbers[-1] if numbers else ""

def evaluate(model, tokenizer, dataset, n_samples=100):
    model.eval()
    correct = 0
    total = min(n_samples, len(dataset))

    for i in range(total):
        example = dataset[i]
        prompt = f"Question: {example['question']}\n\nStep-by-step solution:\n"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        pred = extract_number(generated)
        true = extract_number(example["answer"])

        if pred == true:
            correct += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    train_ds, eval_ds = load_gsm8k()
    tokenizer = build_tokenizer()
    model = load_model()

    trainer = train(train_ds, eval_ds, model, tokenizer)

    evaluate(model, tokenizer, eval_ds)