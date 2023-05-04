# AutoFix

Static Analysis + LLM = AutoFix

Update on 4th May 2023:
- We now support using the [CodeGen2](https://github.com/salesforce/CodeGen2) model from Salesforce. Just use `--model Salesforce/codegen2-1B` with AutoFix.

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
