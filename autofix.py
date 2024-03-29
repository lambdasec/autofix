from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from transformers import pipeline
import os
import torch
import sys
import json
import argparse


HF_TOKEN = os.environ.get("HF_TOKEN")
FIM_PREFIX = "<fim-prefix>"
FIM_MIDDLE = "<fim-middle>"
FIM_SUFFIX = "<fim-suffix>"
FIM_PAD = "<fim-pad>"
EOD = "<|endoftext|>"
MASK_1 = "<mask_1>"
SEP = "<sep>"
EOM = "<eom>"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def format(prefix, suffix):
  return prefix + MASK_1 + suffix + EOD + SEP + MASK_1

def post_processing_fim(prefix, middle, suffix):
    return f"{prefix}{middle}{suffix}"

def fim_generation(model, tokenizer_fim, prompt, max_new_tokens, temperature):
    prefix = prompt.split("<FILL-HERE>")[0]
    suffix = prompt.split("<FILL-HERE>")[1]
    if opt.model is None or opt.model.startswith("bigcode") or opt.model == "lambdasec/santafixer":
        list_of_middles = infill(model, tokenizer_fim, (prefix, suffix), max_new_tokens, temperature)
        # for middle in list_of_middles:
        #     print("\n<options>\n")
        #     print(middle)
        # [middle] = [list_of_middles[0]]
        return [post_processing_fim(prefix, middle, suffix) for middle in list_of_middles]
    else:
        text = format(prefix, suffix)
        # input_ids = tokenizer(text, return_tensors="pt").input_ids
        # generated_ids = model.generate(input_ids, max_length=128)
        inputs = tokenizer_fim(text, return_tensors="pt", padding=True, return_token_type_ids=False,
                           max_length=1024, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=1,
                num_return_sequences=10,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer_fim.eos_token_id
            )
        # print(tokenizer_fim.decode(generated_ids[0], skip_special_tokens=False)[len(text):])
        pats = [r"\n\n^#", "^'''", "\n\n\n"]
        list_of_middles = [extract_mask(tokenizer_fim.decode(tensor, skip_special_tokens=False, truncate_before_pattern=pats), text) for tensor in outputs]
        return [post_processing_fim(prefix, middle, suffix) for middle in list_of_middles]

def extract_mask(s: str, text: str):
    if EOM not in s:
        print("*** File truncated ***")  
    start = len(text)
    stop = s.find(EOM, start) or len(s)
    # print(s)
    return s[start:stop]  

def extract_fim_part(s: str):
    # Find the index of 
    # print(s)
    if FIM_MIDDLE not in s: 
        print("*** File truncated ***")
    start = s.find(FIM_MIDDLE) + len(FIM_MIDDLE)
    stop = s.find(EOD, start) or len(s)
    return s[start:stop]

def infill(model, tokenizer_fim, prefix_suffix_tuples, max_new_tokens, temperature):
    if type(prefix_suffix_tuples) == tuple:
        prefix_suffix_tuples = [prefix_suffix_tuples]
        
    prompts = [f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}" for prefix, suffix in prefix_suffix_tuples]
    # `return_token_type_ids=False` is essential, or we get nonsense output.
    inputs = tokenizer_fim(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                           max_length=1024, truncation=True).to(device)
    end_sequence = tokenizer_fim.encode('\n', return_tensors='pt')[0]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=1,
            num_return_sequences=10,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer_fim.eos_token_id,
            eos_token_id=end_sequence
        )
    # WARNING: cannot use skip_special_tokens, because it blows away the FIM special tokens.
    return [        
        extract_fim_part(tokenizer_fim.decode(tensor, skip_special_tokens=False)) for tensor in outputs
    ]

def code_generation(prompt, max_new_tokens, temperature, model, tokenizer_fim):
    #set_seed(seed)
    
    if "<FILL-HERE>" in prompt:
        # print('<Original Model Output here:>\n' + fim_generation(original_model,original_tokenizer_fim, prompt, max_new_tokens, temperature))
        return fim_generation(model, tokenizer_fim, prompt, max_new_tokens, temperature)
    else:
        print("No infilling token found, please add the token <FILL-HERE> in the code at the place you want the model to do infilling")

def process():
    max_new_tokens = 32
    temperature = 0.8
    input_file = opt.input
    tmp_prompt_file = input_file.split('.')[0] + '_prompt.' + input_file.split('.')[1]
    tmp_file = 'results.json'
    output_file = input_file.split('.')[0] + '_fixed.' + input_file.split('.')[1]
    if input_file.endswith(".java"):
        config_str = 'java'
        comment_str = '//'
    elif input_file.endswith(".js"):
        config_str = 'javascript'
        comment_str = '//'
    elif input_file.endswith(".py"):
        config_str = 'python'
        comment_str = '#'
    else:
        print("Only .java, .js and .py files are supported as input")
        exit(1)
    scan_command_input = "semgrep --config p/"+ config_str +" "+ input_file +" --output "+ tmp_file +" --json > /dev/null 2>&1"
    scan_command_output = "semgrep --config p/"+ config_str +" "+ output_file +" --output "+ tmp_file +" --json > /dev/null 2>&1"
    if os.path.isfile(input_file):
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        print("Scanning file " + input_file + "...")
        os.system(scan_command_input)
        with open(tmp_file, 'r') as jf:
            data = json.load(jf)
        if len(data["errors"]) == 0:
            if len(data["results"]) == 0:
                print(input_file + " has no vulnerabilities")
                exit(0)
            else:
                print("Vulnerability found in " + input_file + "...")
                cwe = data["results"][0]["extra"]["metadata"]["cwe"][0]
                lines = data["results"][0]["extra"]["lines"]
                with open(input_file, 'r') as rf:
                    file_content = rf.read()
                prefix = file_content.split(lines)[0]
                suffix = file_content.split(lines)[1]
                write_content = prefix + '\n' + comment_str + ' BUG: ' + cwe + '\n' + comment_str + lines + '\n' + comment_str + ' FIXED: \n<FILL-HERE>\n' + suffix
                with open(tmp_prompt_file, 'w') as wf:
                    wf.write(write_content)
                print("Attempting fix with prompt file " + tmp_prompt_file + "...")
                model = "lambdasec/santafixer" if opt.model is None else opt.model
                tokenizer_fim = AutoTokenizer.from_pretrained(model, trust_remote_code=True, padding_side="left", use_auth_token=HF_TOKEN)
                
                if model == "lambdasec/santafixer" or model == "bigcode/santacoder":
                    tokenizer_fim.add_special_tokens({
                        "additional_special_tokens": [EOD, FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD]
                    })
                    
                tokenizer_fim.add_special_tokens({
                    "pad_token": EOD
                })
                
                model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True, use_auth_token=HF_TOKEN).to(device)
                
                with open(tmp_prompt_file, 'r') as rf:
                    s = rf.read()
                infill_token_id = s.find("<FILL-HERE>")
                prefix_index = infill_token_id - 1600
                if prefix_index < 0:
                    prefix_index = 0 
                suffix_index = infill_token_id + 1600
                if suffix_index + 1600 > len(s):
                    suffix_index = len(s)
                text = s[prefix_index:suffix_index]
                generations = code_generation(text, max_new_tokens, temperature, model, tokenizer_fim)
                i = 0
                for fixed_code in generations:
                    i += 1
                    fixed_code = s[:prefix_index] + fixed_code + s[suffix_index:]
                    if os.path.exists(output_file):
                        os.remove(output_file)
                    with open(output_file, 'w') as wf:
                        wf.write(fixed_code)
                    if os.path.exists(tmp_file):
                        os.remove(tmp_file)
                    os.system(scan_command_output)
                    with open(tmp_file, 'r') as jf:
                        data = json.load(jf)
                    if len(data["errors"]) == 0 and len(data["results"]) == 0:
                        print("\n Auto fixed file " + output_file + " with code generated at attempt " + str(i))
                        break
                if i == 10:
                    print("Auto fix couldn't fix the file " + input_file)
        else:
            print(input_file + " has parsing errors")
            exit(3)
    else:
        print(input_file + " is not a valid file")
        exit(2)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Specify the Hugging Face model')
    parser.add_argument('--input', type=str, help='The file to scan and fix')
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')
    opt = parser.parse_args()
    # print(opt)
    if opt.input is None:
        print('No input file specified, use --input filename to scan and fix')
    else:
        if opt.model is None:
            print('No model is specified, using lambdasec/santafixer (see https://huggingface.co/lambdasec/santafixer)')
        process()