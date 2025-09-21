import os
import streamlit as st
from typing import List, Dict, Any

# API Configuration - HuggingFace Only
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co"

def get_huggingface_token():
    """Get HuggingFace API token"""
    token = None
    try:
        token = st.secrets["HUGGINGFACE_TOKEN"]
        if token and len(token) > 10:
            return token
    except Exception:
        pass
    
    token = os.getenv("HUGGINGFACE_TOKEN", "")
    if token and len(token) > 10:
        return token
    
    return ""

HUGGINGFACE_TOKEN = get_huggingface_token()

# ✅ HUGGINGFACE MODELS - Free Inference API
# Best free models available on HuggingFace Inference
HUGGINGFACE_MODELS = {
    # Llama models (Meta) - Excellent for all tasks
    "llama_7b": "microsoft/DialoGPT-medium",  # Good for chat
    "llama_13b": "meta-llama/Llama-2-13b-chat-hf",  # Best quality (if available)
    
    # Mistral models (Mistral AI) - Fast and capable
    "mistral_7b": "mistralai/Mistral-7B-Instruct-v0.1",  # Excellent instruction following
    "mixtral_8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",  # Very powerful
    
    # CodeLlama (Meta) - Good for structured tasks
    "codellama_7b": "codellama/CodeLlama-7b-Instruct-hf",  # Good for JSON extraction
    "codellama_13b": "codellama/CodeLlama-13b-Instruct-hf",
    
    # Alternative reliable models
    "flan_t5_xl": "google/flan-t5-xl",  # Reliable for structured tasks
    "flan_ul2": "google/flan-ul2",  # Very capable
    
    # Backup models (always available)
    "gpt2_medium": "gpt2-medium",  # Reliable fallback
    "distilbert": "distilbert-base-uncased",  # Lightweight
}

# Model selection - Best free models for different tasks
DATA_EXTRACTION_MODEL = HUGGINGFACE_MODELS["mistral_7b"]     # Mistral for structured data
RELEVANCE_SCORING_MODEL = HUGGINGFACE_MODELS["mistral_7b"]   # Mistral for analysis
CHATBOT_MODEL = HUGGINGFACE_MODELS["mistral_7b"]            # Mistral for conversations
MAIN_MODEL = HUGGINGFACE_MODELS["mistral_7b"]               # Primary model

# Fallback models (if primary models fail)
FALLBACK_MODELS = [
    HUGGINGFACE_MODELS["flan_t5_xl"],
    HUGGINGFACE_MODELS["gpt2_medium"]
]

# File processing settings
UPLOAD_FOLDER = "./data/uploads"
CACHE_PATH = "./data/hf_cache"
MAX_FILE_SIZE = 15 * 1024 * 1024  # 15MB

# Scoring configuration - Optimized for HuggingFace models
SCORING_WEIGHTS = {
    "relevance_score": 0.65,     # Primary: LLM relevance scoring
    "experience_match": 0.25,    # Experience alignment 
    "skills_match": 0.10         # Skills extraction and matching
}

# HuggingFace API settings - optimized for free tier
HUGGINGFACE_SETTINGS = {
    "data_extraction": {
        "max_length": 1000,      # Conservative for free tier
        "temperature": 0.1,      # Focused for data extraction
        "do_sample": True,
        "top_p": 0.8,
        "timeout": 60,           # Longer timeout for free tier
        "wait_for_model": True,  # Important for free tier
        "max_retries": 3
    },
    "relevance_scoring": {
        "max_length": 500,       # Shorter for scoring
        "temperature": 0.3,      # Slightly more creative
        "do_sample": True,
        "top_p": 0.9,
        "timeout": 45,
        "wait_for_model": True,
        "max_retries": 3
    },
    "chatbot": {
        "max_length": 300,       # Conversational length
        "temperature": 0.5,      # More creative for chat
        "do_sample": True,
        "top_p": 0.95,
        "timeout": 30,
        "wait_for_model": True,
        "max_retries": 2
    }
}

# Thresholds
MIN_SCORE_THRESHOLD = 0.3
TOP_CANDIDATES = 15

# Job templates
JOB_TEMPLATES = {
    "Data Scientist": """
    Data Scientist position requiring Python, SQL, machine learning experience with 
    TensorFlow/PyTorch, statistical analysis skills, 3+ years experience, 
    Master's degree preferred, cloud platforms knowledge, strong problem-solving abilities.
    """,
    
    "Software Engineer": """
    Software Engineer role requiring programming skills in Python/JavaScript/Java,
    web development frameworks, database knowledge, 2+ years experience,
    Computer Science degree, version control (Git), agile methodologies experience.
    """,
    
    "Marketing Manager": """
    Marketing Manager position requiring 5+ years marketing experience, team leadership,
    digital marketing expertise (SEO, SEM, social media), analytics tools experience,
    campaign management, Bachelor's degree in Marketing/Business, excellent communication skills.
    """,
    
    "Product Manager": """
    Product Manager role requiring 3+ years product management experience,
    product lifecycle management, analytical skills, Agile/Scrum methodology,
    technical background, market research skills, stakeholder management abilities.
    """
}

# HuggingFace Prompts - Optimized for open-source models
HUGGINGFACE_PROMPTS = {
    "data_extraction": """<s>[INST] You are a resume parser. Extract structured information from this resume.

Resume Text:
{resume_text}

Extract the following as JSON:
{{
    "name": "candidate name",
    "email": "email address",
    "phone": "phone number",
    "experience_years": number,
    "current_role": "job title",
    "skills": ["skill1", "skill2", "skill3"],
    "education": "highest degree",
    "companies": ["company1", "company2"],
    "summary": "brief professional summary"
}}

Respond only with valid JSON. [/INST]""",

    "relevance_scoring": """<s>[INST] You are an HR expert. Score this candidate's fit for the job from 0-100.

Job Requirements:
{job_description}

Candidate Profile:
{candidate_data}

Consider: technical skills (30%), relevant experience (35%), education (15%), career growth (10%), cultural fit (10%).

Provide only a number from 0-100, followed by a brief explanation.

Format: Score: XX
Reason: Brief explanation

[/INST]""",

    "detailed_analysis": """<s>[INST] You are a senior HR consultant. Analyze this candidate for the role.

Job: {job_title}
Requirements: {job_description}

Candidate: {candidate_data}

Provide analysis covering:
1. Key Strengths (3 points)
2. Areas of Concern (2 points)  
3. Interview Recommendations (2 focus areas)
4. Overall Recommendation (Hire/Consider/Pass)

Keep response under 300 words. [/INST]""",

    "experience_analysis": """<s>[INST] Analyze work experience from this candidate data:

{candidate_data}

Provide:
- Total years experience: X
- Career level: Junior/Mid/Senior
- Key companies: List top 2
- Role progression: Brief assessment
- Experience quality: Good/Average/Excellent

[/INST]""",

    "chatbot": """<s>[INST] You are a professional HR assistant. Answer this question about resume screening and hiring.

{context}

Question: {question}

Provide helpful, professional advice in 2-3 sentences. [/INST]"""
}

# Directory creation
def create_directories():
    """Create necessary directories"""
    directories = [
        UPLOAD_FOLDER,
        CACHE_PATH,
        "./data/temp_uploads",
        "./logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Model information
def get_model_info():
    """Get model information"""
    return {
        "deployment_type": "🤗 HuggingFace Only (Free)",
        "main_model": {
            "name": "Mistral 7B Instruct",
            "full_name": MAIN_MODEL,
            "provider": "HuggingFace Inference API",
            "type": "Open Source LLM",
            "cost": "FREE (with rate limits)",
            "status": "✅ HF Ready" if HUGGINGFACE_TOKEN else "❌ No Token"
        },
        "capabilities": {
            "data_extraction": DATA_EXTRACTION_MODEL.split('/')[-1],
            "relevance_scoring": RELEVANCE_SCORING_MODEL.split('/')[-1],
            "detailed_analysis": MAIN_MODEL.split('/')[-1],
            "chatbot": CHATBOT_MODEL.split('/')[-1]
        },
        "advantages": [
            "Completely FREE with HuggingFace account",
            "Open-source models (Mistral, Llama, etc.)",
            "No usage costs, only rate limits",
            "Privacy-focused (can run locally later)",
            "Community-driven model improvements"
        ]
    }

# Validation
def validate_setup():
    """Validate HuggingFace setup"""
    issues = []
    
    if not HUGGINGFACE_TOKEN:
        issues.append("❌ Missing HuggingFace API token")
        issues.append("🔧 Add HUGGINGFACE_TOKEN to .streamlit/secrets.toml")
        issues.append("📝 Get free token from https://huggingface.co/settings/tokens")
    elif len(HUGGINGFACE_TOKEN) < 20:
        issues.append("❌ Invalid HuggingFace token format")
    else:
        issues.append("✅ HuggingFace API configured")
        issues.append("✅ Using Mistral 7B for all tasks")
        issues.append("✅ FREE tier with rate limits")
        issues.append("🎉 Zero-cost resume screening!")
    
    return issues

# Utility functions
def get_huggingface_prompt(template_key: str, **kwargs) -> str:
    """Get formatted HuggingFace prompt"""
    template = HUGGINGFACE_PROMPTS.get(template_key, "")
    return template.format(**kwargs)

# API Health Check
def check_api_health():
    """Check HuggingFace API health"""
    health = {
        "huggingface": {
            "available": bool(HUGGINGFACE_TOKEN),
            "models": {
                "data_extraction": DATA_EXTRACTION_MODEL,
                "scoring": RELEVANCE_SCORING_MODEL,
                "chatbot": CHATBOT_MODEL
            },
            "purpose": "All resume screening tasks (FREE)",
            "rate_limits": "Yes - HuggingFace free tier"
        },
        "overall_status": "ready" if HUGGINGFACE_TOKEN else "incomplete",
        "architecture": "HuggingFace Only - Free & Open Source"
    }
    
    return health

# Rate limiting helpers
def get_rate_limit_info():
    """Get rate limiting information"""
    return {
        "free_tier": "1000 requests/month per model",
        "rate_limit": "~10 requests/minute per model",
        "recommendations": [
            "Use caching aggressively",
            "Process in smaller batches", 
            "Consider Pro subscription for higher limits",
            "Rotate between different models if needed"
        ],
        "cost": "FREE with limits, $9/month for Pro"
    }

# Model selection helper
def select_best_available_model(task_type: str) -> str:
    """Select best available model for task"""
    task_models = {
        "data_extraction": [DATA_EXTRACTION_MODEL] + FALLBACK_MODELS,
        "scoring": [RELEVANCE_SCORING_MODEL] + FALLBACK_MODELS,
        "chatbot": [CHATBOT_MODEL] + FALLBACK_MODELS,
        "analysis": [MAIN_MODEL] + FALLBACK_MODELS
    }
    
    return task_models.get(task_type, FALLBACK_MODELS)[0]

# Prompt optimization for free models
def optimize_prompt_for_free_tier(prompt: str, max_length: int = 1000) -> str:
    """Optimize prompts for free tier limits"""
    if len(prompt) > max_length:
        # Truncate while preserving structure
        lines = prompt.split('\n')
        truncated = []
        current_length = 0
        
        for line in lines:
            if current_length + len(line) < max_length - 100:  # Leave buffer
                truncated.append(line)
                current_length += len(line)
            else:
                truncated.append("... (truncated for free tier limits)")
                break
        
        return '\n'.join(truncated)
    
    return prompt