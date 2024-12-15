import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['HF_HOME'] = "hf_cache"


import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoTokenizer
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer

from tqdm import tqdm

import argparse


parser = argparse.ArgumentParser(description='IndicTrans Inference')
parser.add_argument('--src', type=str, help='Source Language')
parser.add_argument('--tgt', type=str, help='Target Language')
parser.add_argument("--input", "-i", type=str, help="Input File")
parser.add_argument("--output", "-o", type=str, help="Output File")
parser.add_argument("--batch", "-b", type=int, default=8, help="Batch Size")

args = parser.parse_args()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
quantization = None


def initialize_model_and_tokenizer(ckpt_dir, direction, quantization):
    if quantization == "4-bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8-bit":
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        qconfig = None

    tokenizer = IndicTransTokenizer(direction=direction)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
    )

    if qconfig == None:
        model = model.to(DEVICE)
        if DEVICE == "cuda":
            model.half()

    model.eval()

    return tokenizer, model


def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]

        # Preprocess the batch and extract entity mappings
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

        # Tokenize the batch and generate input encodings
        inputs = tokenizer(
            batch,
            src=True,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)

        # Generate translations using the model
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        # Decode the generated tokens into text
        generated_tokens = tokenizer.batch_decode(generated_tokens.detach().cpu().tolist(), src=False)

        # Postprocess the translations, including entity replacement
        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

        del inputs
        torch.cuda.empty_cache()

    return translations


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

# model_name = "ai4bharat/indictrans2-indic-indic-dist-320M"
model_name = "ai4bharat/indictrans2-indic-indic-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True).to(DEVICE)

ip = IndicProcessor(inference=True)





def translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    translations = []
    for i in tqdm(range(0, len(input_sentences), BATCH_SIZE) , desc="Translating"):
        limit = min(i + BATCH_SIZE, len(input_sentences))
        batch = input_sentences[i : limit]

        # Preprocess the batch and extract entity mappings
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)


        # Tokenize the batch and generate input encodings
        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)


        # Generate translations using the model
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )


        # # Decode the generated tokens into text
        with tokenizer.as_target_tokenizer():
            generated_tokens = tokenizer.batch_decode(
                generated_tokens.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

        # Postprocess the translations, including entity replacement
        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

        del inputs
        torch.cuda.empty_cache()

    return translations



BATCH_SIZE = args.batch

src_lang = args.src

tgt_lang = args.tgt

input_file = args.input

output_file = args.output

with open(input_file, "r") as f:
    input_sentences = [x.strip() for x in f.readlines()]
print("Starting Translation")
translations = translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip)

with open(output_file, "w") as f:
    for translation in translations:
        f.write(translation + "\n")

