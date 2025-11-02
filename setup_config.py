#!/usr/bin/env python3
"""
Configuration setup for the Lean Theorem Prover Agent
Run this to set up your API keys and configuration
"""

import os
from pathlib import Path

def setup_config():
    # Create .env file
    env_content = """# ModelScope API Configuration (for Planning and Verification)
MODELSCOPE_API_KEY=your_modelscope_token_here

# Alibaba Cloud DashScope API Configuration (for Generation)
DASHSCOPE_API_KEY=your_dashscope_api_key_here
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# Model Configuration  
PLANNING_MODEL=deepseek-ai/DeepSeek-V3.1
GENERATION_MODEL=qwen3-coder-plus
VERIFICATION_MODEL=deepseek-ai/DeepSeek-R1-0528

# RAG Configuration
MAX_CHUNKS=10
CHUNK_SIZE=1000
OVERLAP_SIZE=200
MAX_RETRIES=3
RETRY_DELAY=1.0

# Lean Configuration
LEAN_TIMEOUT=300
MAX_PROOF_ATTEMPTS=5
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("Created .env file")
    print("Please edit .env and set:")
    print("  1. MODELSCOPE_API_KEY - Your ModelScope Token")
    print("  2. DASHSCOPE_API_KEY - Your Alibaba Cloud DashScope API Key")
    
    # Create requirements.txt
    requirements = """openai>=1.0.0
python-dotenv>=1.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
tiktoken>=0.5.0
nltk>=3.8.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("Created requirements.txt")
    print("\nNext steps:")
    print("1. Edit the .env file with your API key")
    print("2. Install requirements: pip install -r requirements.txt")

if __name__ == "__main__":
    setup_config()