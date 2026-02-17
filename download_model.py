#!/usr/bin/env python
"""Download HuggingFace model with SSL verification disabled."""

import os
import ssl
import warnings

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = 'EleutherAI/pythia-160m'

print(f"Downloading {model_name}...")
print("This may take a few minutes...")

try:
    print("\n1. Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("   ✓ Tokenizer downloaded successfully")
    
    print("\n2. Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print("   ✓ Model downloaded successfully")
    
    print(f"\n✓ Model cached and ready to use!")
    print(f"  Model: {model_name}")
    print(f"  Parameters: {model.num_parameters():,}")
    
except Exception as e:
    print(f"\n✗ Error downloading model: {e}")
    raise
