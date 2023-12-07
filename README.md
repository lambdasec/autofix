# AutoFix

Static Analysis + LLM = AutoFix

_Note: If you are looking for a cloud service for vulnerability remediation, please try [patched](https://www.patched.codes/)._

- The new [StarCoder](https://huggingface.co/bigcode/starcoderbase-1b) model is now supported. Pass `--model bigcode/starcoderbase-1b` to AutoFix to try the 1B parameter base model. 

- We now support using the [CodeGen2](https://github.com/salesforce/CodeGen2) model from Salesforce. Just use `--model Salesforce/codegen2-1B` with AutoFix. Note that the inference on CPU with `CodeGen2` is very slow compared to `SantaFixer`.

In the initial release, we used Semgrep for doing static analysis and the [SantaFixer](https://huggingface.co/lambdasec/santafixer) LLM for bug fixing.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```
python autofix.py --input examples/example.java
```

## Demo

![](https://github.com/lambdasec/autofix/blob/main/demo.gif)

## How it works?
![](https://github.com/lambdasec/autofix/blob/main/howitworks.png)
