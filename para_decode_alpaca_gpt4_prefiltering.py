# pip install bitsandbytes accelerate
import random
import time
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GemmaForCausalLM, BitsAndBytesConfig
from transformers.generation.stopping_criteria import StoppingCriteriaList, EosTokenCriteria
from transformers.cache_utils import Cache
import json
import types
import argparse

# Define the argument parser
parser = argparse.ArgumentParser(description="Hyper-parameters for the model")

parser.add_argument('--model', type=str, default='google/gemma-7b-it', help='Path to the original model')
parser.add_argument('--assist_model', type=str, default='google/gemma-2b-it', help='Path to the assist model')
parser.add_argument('--max_new_tokens', type=int, default=128, help='Maximum number of new tokens')
parser.add_argument('--top_k', type=int, default=100, help='Top K tokens to consider')
parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for sampling')
parser.add_argument('--print_token_details', action='store_true', help='Whether to print token details')

parser.add_argument('--data_path', type=str, default='../GPT-4-LLM/data/', help='Path to the data directory')
parser.add_argument('--data_filename', type=str, nargs='+', default=["alpaca_gpt4_data.json", "alpaca_gpt4_data_zh.json"], help='List of data filenames')

parser.add_argument('--rand_seed', type=int, default=0, help='Random seed')
parser.add_argument('--lang', type=str, default='en', choices=['en', 'zh', 'en&zh'], help='Language')
parser.add_argument('--dataset_size', type=int, default=20, help='Size of the dataset')
parser.add_argument('--batch_size', type=int, default=2, help='Number of inputs at the same time')

parser.add_argument('--assist_token_conf_threshold', type=float, default=0.99, help='Confidence threshold for assist model tokens')
parser.add_argument('--compressed_token_conf_threshold', type=float, default=5, help='Confidence threshold for compressed model tokens')
# 1 for batch size 1
# 0.05 for batch size 2, 0.085 for batch size 3, ...
# the threshold is learnt from linear regression on training data
parser.add_argument('--overlap_ratio_threshold', type=float, default=0.05, help='Overlap ratio threshold for top-K tokens')

parser.add_argument('--output_json', type=str, default='output.json', help='Output JSON file')

# Parse the arguments
args = parser.parse_args()

# Set the random seeds
random.seed(args.rand_seed)
torch.manual_seed(args.rand_seed)

# Assign parsed arguments to the original variables
MODEL = args.model
ASSIST_MODEL = args.assist_model
MAX_NEW_TOKENS = args.max_new_tokens
TOP_K = args.top_k
TEMPERATURE = args.temperature
PRINT_TOKEN_DETAILS = args.print_token_details

DATA_PATH = args.data_path
DATA_FILENAME = args.data_filename

RAND_SEED = args.rand_seed
LANG = args.lang
DATASET_SIZE = args.dataset_size
INPUT_NUM = args.batch_size

ASSIST_TOKEN_CONF_THRESHOLD = args.assist_token_conf_threshold
COMPRESSED_TOKEN_CONF_THRESHOLD = args.compressed_token_conf_threshold
OVERLAP_RATIO_THRESHOLD = args.overlap_ratio_threshold

# Update the data filenames based on the language
if LANG == 'en':
    DATA_FILENAME = [DATA_FILENAME[0]]
elif LANG == 'zh':
    DATA_FILENAME = [DATA_FILENAME[1]]
elif LANG == 'en&zh':
    DATA_FILENAME = DATA_FILENAME

OUTPUT_FILE = args.output_json

# Print the variables to verify
print(f"MODEL: {MODEL}")
print(f"ASSIST_MODEL: {ASSIST_MODEL}")
print(f"MAX_NEW_TOKENS: {MAX_NEW_TOKENS}")
print(f"TOP_K: {TOP_K}")
print(f"TEMPERATURE: {TEMPERATURE}")
print(f"PRINT_TOKEN_DETAILS: {PRINT_TOKEN_DETAILS}")
print(f"DATA_PATH: {DATA_PATH}")
print(f"DATA_FILENAME: {DATA_FILENAME}")
print(f"RAND_SEED: {RAND_SEED}")
print(f"LANG: {LANG}")
print(f"DATASET_SIZE: {DATASET_SIZE}")
print(f"INPUT_NUM: {INPUT_NUM}")
print(f"ASSIST_TOKEN_CONF_THRESHOLD: {ASSIST_TOKEN_CONF_THRESHOLD}")
print(f"COMPRESSED_TOKEN_CONF_THRESHOLD: {COMPRESSED_TOKEN_CONF_THRESHOLD}")
print(f"OVERLAP_RATIO_THRESHOLD: {OVERLAP_RATIO_THRESHOLD}")
print(f"OUTPUT_FILE: {OUTPUT_FILE}")
print(flush=True)

def make_prompt(inst, input):
    return f"{inst}\n\n{input}\n"

def read_data(data_path, data_filename):
    with open(data_path + data_filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    ref_list = []
    for sample in data:
        ref_list.append([{ "role": "user", "content": make_prompt(sample["instruction"], sample["input"])}])
    return ref_list

chats = [read_data(DATA_PATH, filename) for filename in DATA_FILENAME]
chats = [chat for sublist in chats for chat in sublist]
random.shuffle(chats)
chats = chats[:DATASET_SIZE*INPUT_NUM]

# format chats in sublist with length of INPUT_NUM
chats = [chats[i:i+INPUT_NUM] for i in range(0, len(chats), INPUT_NUM)]

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", torch_dtype=torch.float16).eval() #, attn_implementation="flash_attention_2")

def prepare_inputs_for_generation(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    cache_position=None,
    use_cache=True,
    **kwargs,
):
    past_length = 0
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
            max_cache_length = (
                torch.tensor(past_key_values.get_max_length(), device=input_ids.device if input_ids is not None else inputs_embeds.device)
                if past_key_values.get_max_length() is not None
                else None
            )
            cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)
        # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
        else:
            cache_length = past_length = past_key_values[0][0].shape[2]
            max_cache_length = None

        if input_ids is not None:
            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
        else:
            if attention_mask is not None and attention_mask.shape[1] > inputs_embeds.shape[1]:
                inputs_embeds = inputs_embeds[:, -(attention_mask.shape[1] - past_length) :]
            elif past_length < inputs_embeds.shape[1]:
                inputs_embeds = inputs_embeds[:, past_length:]

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + (input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]) > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -(input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]) :]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
        # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
        # TODO: use `next_tokens` directly instead.
        if input_ids is not None:
            model_inputs = {"input_ids": input_ids.contiguous()}
        else:
            model_inputs = {"inputs_embeds": inputs_embeds}

    input_length = position_ids.shape[-1] if position_ids is not None else (input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2])
    if cache_position is None:
        cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device if input_ids is not None else inputs_embeds.device)
    elif use_cache:
        cache_position = cache_position[-input_length:]

    model_inputs.update(
        {
            "position_ids": position_ids,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
        }
    )
    return model_inputs
model.prepare_inputs_for_generation = types.MethodType(prepare_inputs_for_generation, model)

quantization_config = BitsAndBytesConfig(load_in_4bit=True)
assist_tokenizer = AutoTokenizer.from_pretrained(ASSIST_MODEL) # identical to tokenizer
assist_model = AutoModelForCausalLM.from_pretrained(ASSIST_MODEL, device_map="auto", torch_dtype=torch.float16).eval() # quantization_config=quantization_config, 


def logits_to_probs(in_logits, temperature: float = 1.0, top_k: Optional[int] = None):
    in_logits = in_logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(in_logits, min(top_k, in_logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        in_logits = torch.where(in_logits < pivot, -float("Inf"), in_logits)
    probs = torch.nn.functional.softmax(in_logits, dim=-1)
    return probs

def output_last_logits(last_token_logits, compressed_logits=None, actual_input_ids=None, model_kwargs=None, aid=None, temperature=TEMPERATURE):
    if compressed_logits is not None:
        last_token_probs = logits_to_probs(last_token_logits, top_k=TOP_K)
        if torch.max(last_token_probs).item() < ASSIST_TOKEN_CONF_THRESHOLD:
            compressed_probs = logits_to_probs(compressed_logits)
            if torch.max(compressed_probs).item() >= COMPRESSED_TOKEN_CONF_THRESHOLD:
                last_token_probs = logits_to_probs(compressed_logits)
            else:
                last_token_probs = last_token_probs * compressed_probs
            # TOP-K
            v, _ = torch.topk(last_token_probs, min(TOP_K, last_token_probs.size(-1)))
            pivot = v.select(-1, -1).unsqueeze(-1)
            last_token_probs = torch.where(last_token_probs < pivot, 0, last_token_probs)
        
        positive_probs_indices = last_token_probs.nonzero().squeeze()
        if len(positive_probs_indices) == 0:
            model_inputs = model.prepare_inputs_for_generation(actual_input_ids, **model_kwargs[aid])
            model_outputs = model.forward(**model_inputs, return_dict=True)
            actual_logits = model_outputs.logits[:, -1]
            last_token_probs = logits_to_probs(actual_logits, temperature=TEMPERATURE)
            model_kwargs[aid] = model._update_model_kwargs_for_generation(model_outputs, model_kwargs[aid])        
        if temperature == 0.0:
            return torch.argmax(last_token_probs)
        return torch.multinomial(last_token_probs, num_samples=1).squeeze(1)
    else:
        if temperature == 0.0:
            return torch.argmax(last_token_logits, dim=-1)
        last_token_probs = logits_to_probs(last_token_logits, temperature=TEMPERATURE, top_k=TOP_K)
        return torch.multinomial(last_token_probs, num_samples=1).squeeze(1)



if __name__ == "__main__":

    compressed_gen_time = 0
    original_gen_time = 0
    assist_model_gen_time, assist_model_batchgen_time = 0, 0

    output_json_list = {
        "compressed_gen": [],
        "original_gen": [],
        "assist_model_gen": []
    }
    with torch.no_grad():

        for sub_chats in chats:

            inputs_list = []
            for cid, chat in enumerate(sub_chats):
                prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

                inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
                inputs_list.append(inputs)

            # sort inputs_list by length
            inputs_list = sorted(inputs_list, key=lambda x: x.shape[1], reverse=True)
            longest_len = inputs_list[0].shape[1]

            # generate until max length
            for iid in range(1, len(inputs_list)):
                if inputs_list[iid].shape[1] < longest_len:
                    inputs_list[iid] = model.generate(input_ids=inputs_list[iid], max_new_tokens=longest_len-inputs_list[iid].shape[1], do_sample=False)
            inputs_list = [inputs for inputs in inputs_list if inputs.shape[1] == longest_len]
            origin_inputs_list = inputs_list.copy()

            # COMPRESSSED GENERATION
            # print(inputs_list)
            # print(flush=True)
            compressed_gen_start = time.time()

            inputs_all = torch.cat(inputs_list, dim=0)
            generation_config, compressed_model_kwargs = model._prepare_generation_config(None)
            model_kwargs = [compressed_model_kwargs.copy() for _ in inputs_list]
            assist_generation_config, assist_model_kwargs = assist_model._prepare_generation_config(None)
            this_peer_finished = False
            unfinished_sequences = torch.ones(inputs_all.shape[0], dtype=torch.long, device=model.device)
            pad_token_id = generation_config.eos_token_id

            compressed_embed = torch.mean(model.get_input_embeddings()(inputs_all), dim=0).unsqueeze(0)

            stopping_criteria = StoppingCriteriaList()
            stopping_criteria.append(EosTokenCriteria(eos_token_id=generation_config.eos_token_id))

            for tid in range(MAX_NEW_TOKENS):

                assist_model_inputs = assist_model.prepare_inputs_for_generation(inputs_all, **assist_model_kwargs)
                assist_model_outputs = assist_model.forward(**assist_model_inputs, return_dict=True)
                assist_last_token_logits_all = assist_model_outputs.logits[:, -1]
                assist_output_tokens = output_last_logits(assist_last_token_logits_all)

                assist_output_tokens = assist_output_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

                assist_model_kwargs = assist_model._update_model_kwargs_for_generation(assist_model_outputs, assist_model_kwargs)

                prefiltered_indices = [iid for iid, assist_last_token_logits in enumerate(assist_last_token_logits_all) if torch.max(logits_to_probs(assist_last_token_logits, top_k=TOP_K)).item() >= ASSIST_TOKEN_CONF_THRESHOLD]
                unfiltered_indices = [iid for iid in range(len(assist_last_token_logits_all)) if iid not in prefiltered_indices]
                prefiltered_inputs_all = inputs_all[prefiltered_indices]
                prefiltered_assist_output_tokens = assist_output_tokens[prefiltered_indices]
                prefiltered_inputs_all = torch.cat([prefiltered_inputs_all, prefiltered_assist_output_tokens[:, None]], dim=-1)
                unfiltered_inputs_all = inputs_all[unfiltered_indices]
                unfiltered_assist_output_tokens = assist_output_tokens[unfiltered_indices]
                
                # Expand inputs_all with pad tokens
                inputs_all = torch.cat([inputs_all, pad_token_id * torch.ones(inputs_all.shape[0], 1, dtype=torch.long, device=model.device)], dim=-1)

                if len(unfiltered_indices) == 0:
                    inputs_all = prefiltered_inputs_all
                    unfinished_sequences = unfinished_sequences & ~stopping_criteria(inputs_all, None)
                    this_peer_finished = unfinished_sequences.max() == 0
                    if this_peer_finished:
                        break
                    continue

                # Get the top-K tokens for each assist_last_token_logits
                top_k_assist_tokens_list = [torch.topk(logits, TOP_K//len(assist_last_token_logits_all)).indices.tolist() for logits in assist_last_token_logits_all]
                all_assist_tokens = [token for sublist in top_k_assist_tokens_list for token in sublist]
                
                inputs_all[prefiltered_indices] = prefiltered_inputs_all

                # Use a set structure to check for duplicate tokens
                if len(unfiltered_indices) == 1 or (len(all_assist_tokens) - len(set(all_assist_tokens))) / len(set(all_assist_tokens)) < OVERLAP_RATIO_THRESHOLD:
                    # use original model for generation
                    actual_last_token_logits_list = []
                    for iid, aid in enumerate(unfiltered_indices):
                        # if unfinished_sequences[aid] == 0:
                        #     continue
                        model_inputs = model.prepare_inputs_for_generation(unfiltered_inputs_all[iid].unsqueeze(0), **model_kwargs[aid])
                        model_outputs = model.forward(**model_inputs, return_dict=True)
                        actual_last_token_logits = model_outputs.logits[:, -1]
                        model_kwargs[aid] = model._update_model_kwargs_for_generation(model_outputs, model_kwargs[aid])

                        actual_last_token_logits_list.append(actual_last_token_logits)
                    
                    # if len(actual_last_token_logits_list) == 0:
                    #     continue
                    unfiltered_next_tok_all = output_last_logits(torch.cat(actual_last_token_logits_list, dim=0))
                    unfiltered_unfinished_sequences = unfinished_sequences[unfiltered_indices]
                    unfiltered_next_tok_all = unfiltered_next_tok_all * unfiltered_unfinished_sequences + pad_token_id * (1 - unfiltered_unfinished_sequences)
                    unfiltered_inputs_all = torch.cat([unfiltered_inputs_all, unfiltered_next_tok_all[:, None]], dim=-1)
                    inputs_all[unfiltered_indices] = unfiltered_inputs_all

                    unfinished_sequences = unfinished_sequences & ~stopping_criteria(inputs_all, None)
                    this_peer_finished = unfinished_sequences.max() == 0
                    if this_peer_finished:
                        break
                    continue

                next_embeds = model.get_input_embeddings()(inputs_all[:, compressed_embed.shape[1]:])
                next_embeds = torch.mean(next_embeds, dim=0)
                compressed_embed = torch.cat([compressed_embed, next_embeds.unsqueeze(0)], dim=1)

                compressed_model_inputs = model.prepare_inputs_for_generation(None, inputs_embeds=compressed_embed, **compressed_model_kwargs)
                compressed_model_outputs = model.forward(**compressed_model_inputs, return_dict=True)
                last_token_logits = compressed_model_outputs.logits[:, -1]
                compressed_model_kwargs = model._update_model_kwargs_for_generation(compressed_model_outputs, compressed_model_kwargs)

                for iid, aid in enumerate(unfiltered_indices):
                    if unfinished_sequences[aid] == 0:
                        continue
                    next_tok = output_last_logits(assist_last_token_logits_all[aid].unsqueeze(0), compressed_logits=last_token_logits, actual_input_ids=unfiltered_inputs_all[iid].unsqueeze(0), model_kwargs=model_kwargs, aid=aid)
                    inputs_all[aid, -1] = next_tok

                unfinished_sequences = unfinished_sequences & ~stopping_criteria(inputs_all, None)
                this_peer_finished = unfinished_sequences.max() == 0
                if this_peer_finished:
                    break

            compressed_gen_end = time.time()
            print("compressed_gen_time: ", compressed_gen_end - compressed_gen_start)
            compressed_gen_time += compressed_gen_end - compressed_gen_start

            print("Compressed Generation:")
            inputs_list = list(torch.unbind(inputs_all, dim=0))
            for iid, inputs in enumerate(inputs_list):
                print(f"Chat {iid}:")
                print(tokenizer.decode(inputs, skip_special_tokens=True, clean_up_tokenization_spaces=False))
                output_json_list["compressed_gen"].append(tokenizer.decode(inputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False))
                print(flush=True)

            # MAIN MODEL GENERATION
            inputs_list = origin_inputs_list.copy()
            # print(inputs_list)
            # print(flush=True)
            outputs_list = []
            original_gen_start = time.time()
            for iid, inputs in enumerate(inputs_list):
                outputs = model.generate(input_ids=inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
                outputs_list.append(outputs)
            original_gen_end = time.time()
            print("original_gen_time: ", original_gen_end - original_gen_start)
            original_gen_time += original_gen_end - original_gen_start

            print("Original Generation:")
            for iid, outputs in enumerate(outputs_list):
                print(f"Chat {iid}:")
                print(tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False))
                output_json_list["original_gen"].append(tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False))
                print(flush=True)

            # ASSIST MODEL GENERATION
            inputs_list = origin_inputs_list.copy()
            # print(inputs_list)
            # print(flush=True)
            outputs_list = []
            assist_model_gen_start = time.time()
            for iid, inputs in enumerate(inputs_list):
                outputs = assist_model.generate(input_ids=inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
                outputs_list.append(outputs)
            assist_model_gen_end = time.time()
            print("assist_model_gen_time: ", assist_model_gen_end - assist_model_gen_start)
            assist_model_gen_time += assist_model_gen_end - assist_model_gen_start
            
            print("Assist Model Generation:")
            for iid, outputs in enumerate(outputs_list):
                print(f"Chat {iid}:")
                print(tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False))
                output_json_list["assist_model_gen"].append(tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False))
                print(flush=True)

            # ASSIST MODEL BATCH GENERATION
            inputs_list = origin_inputs_list.copy()
            # print(inputs_list)
            # print(flush=True)
            assist_model_batchgen_start = time.time()

            inputs_all = torch.cat(inputs_list, dim=0)
            assist_generation_config, assist_model_kwargs = assist_model._prepare_generation_config(None)
            this_peer_finished = False
            unfinished_sequences = torch.ones(inputs_all.shape[0], dtype=torch.long, device=model.device)
            pad_token_id = assist_generation_config.pad_token_id

            stopping_criteria = StoppingCriteriaList()
            stopping_criteria.append(EosTokenCriteria(eos_token_id=assist_generation_config.eos_token_id))

            for tid in range(MAX_NEW_TOKENS):
                assist_model_inputs = assist_model.prepare_inputs_for_generation(inputs_all, **assist_model_kwargs)
                assist_model_outputs = assist_model.forward(**assist_model_inputs, return_dict=True)
                assist_last_token_logits_all = assist_model_outputs.logits[:, -1]
                assist_output_tokens = output_last_logits(assist_last_token_logits_all)

                assist_output_tokens = assist_output_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

                inputs_all = torch.cat([inputs_all, assist_output_tokens[:, None]], dim=-1)
                unfinished_sequences = unfinished_sequences & ~stopping_criteria(inputs_all, None)
                this_peer_finished = unfinished_sequences.max() == 0
                if this_peer_finished:
                    break

                assist_model_kwargs = assist_model._update_model_kwargs_for_generation(assist_model_outputs, assist_model_kwargs)


            assist_model_batchgen_end = time.time()
            print("assist_model_batchgen_time: ", assist_model_batchgen_end - assist_model_batchgen_start)
            assist_model_batchgen_time += assist_model_batchgen_end - assist_model_batchgen_start
            
            print("Assist Model Batch Generation:")
            inputs_list = list(torch.unbind(inputs_all, dim=0))
            for iid, inputs in enumerate(inputs_list):
                print(f"Chat {iid}:")
                print(tokenizer.decode(inputs, skip_special_tokens=True, clean_up_tokenization_spaces=False))
                print(flush=True)

        print("compressed_gen_time: ", compressed_gen_time)
        print("original_gen_time: ", original_gen_time)
        print("assist_model_gen_time: ", assist_model_gen_time)
        print("assist_model_batchgen_time: ", assist_model_batchgen_time)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_json_list, f, indent=4, ensure_ascii=False)
