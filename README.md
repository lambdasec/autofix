# AutoFix

Static Analysis + LLM = AutoFix

In this initial release, we use Semgrep for doing static analysis and the [SantaFixer](https://huggingface.co/lambdasec/santafixer) LLM for bug fixing.

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
