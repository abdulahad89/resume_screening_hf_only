import requests
import streamlit as st
import time
import hashlib
import json
import os
import re
from typing import Dict, List, Any, Optional
from datetime import datetime

class HuggingFaceLLMManager:
    """HuggingFace LLM manager for all resume screening tasks using free inference API"""
    
    def __init__(self):
        from config import (
            HUGGINGFACE_TOKEN, HUGGINGFACE_API_URL, HUGGINGFACE_SETTINGS,
            DATA_EXTRACTION_MODEL, RELEVANCE_SCORING_MODEL, CHATBOT_MODEL, MAIN_MODEL,
            FALLBACK_MODELS
        )
        
        self.api_token = HUGGINGFACE_TOKEN
        self.api_url = HUGGINGFACE_API_URL
        self.settings = HUGGINGFACE_SETTINGS
        self.data_extraction_model = DATA_EXTRACTION_MODEL
        self.relevance_scoring_model = RELEVANCE_SCORING_MODEL
        self.chatbot_model = CHATBOT_MODEL
        self.main_model = MAIN_MODEL
        self.fallback_models = FALLBACK_MODELS
        
        # API headers
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        # Cache settings
        self.cache_dir = "./data/hf_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Performance tracking
        self.stats = {
            'data_extractions': 0,
            'relevance_scores': 0,
            'detailed_analyses': 0,
            'chat_responses': 0,
            'total_api_calls': 0,
            'cache_hits': 0,
            'errors': 0,
            'rate_limit_hits': 0
        }
        
        # Test connection
        self.connection_status = self._test_connection()
    
    def _test_connection(self) -> Dict[str, Any]:
        """Test HuggingFace API connection"""
        status = {
            "connected": False,
            "model_available": False,
            "error": None,
            "rate_limited": False
        }
        
        if not self.api_token:
            status["error"] = "No HuggingFace API token provided"
            return status
        
        try:
            # Simple test with main model
            test_response = self._call_llm_api(
                "Test: Say 'HuggingFace Ready'",
                self.main_model,
                max_length=20,
                temperature=0.1,
                timeout=10
            )
            
            if test_response and "error" not in test_response:
                if isinstance(test_response, list) and len(test_response) > 0:
                    response_text = test_response[0].get('generated_text', '')
                    if 'ready' in response_text.lower():
                        status["connected"] = True
                        status["model_available"] = True
                    else:
                        status["connected"] = True
                        status["model_available"] = True  # Connected but model response unclear
                else:
                    status["error"] = "Unexpected response format"
            elif test_response and "error" in test_response:
                error_msg = test_response.get("error", "Unknown error")
                if "rate limit" in error_msg.lower():
                    status["rate_limited"] = True
                    status["error"] = "Rate limited - try again later"
                else:
                    status["error"] = f"API error: {error_msg}"
            else:
                status["error"] = "No response from HuggingFace API"
                
        except Exception as e:
            status["error"] = f"Connection test failed: {str(e)}"
        
        return status
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive HuggingFace status"""
        return {
            "provider": "HuggingFace Inference API (Free)",
            "models_available": bool(self.api_token),
            "connection_status": self.connection_status,
            "models": {
                "data_extraction": self.data_extraction_model.split('/')[-1],
                "relevance_scoring": self.relevance_scoring_model.split('/')[-1],
                "chatbot": self.chatbot_model.split('/')[-1],
                "main_model": self.main_model.split('/')[-1]
            },
            "api_configured": bool(self.api_token),
            "stats": self.stats.copy(),
            "rate_limits": "Free tier: ~10 requests/minute per model"
        }
    
    def _call_llm_api(self, prompt: str, model: str, max_length: int = 500, 
                      temperature: float = 0.3, timeout: int = 60, max_retries: int = 3) -> Optional[Dict]:
        """Call HuggingFace LLM API with retry logic"""
        
        for attempt in range(max_retries + 1):
            try:
                # Prepare payload
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_length": max_length,
                        "temperature": temperature,
                        "do_sample": True,
                        "top_p": 0.9,
                        "return_full_text": False
                    },
                    "options": {
                        "wait_for_model": True,
                        "use_cache": True
                    }
                }
                
                response = requests.post(
                    f"{self.api_url}/models/{model}",
                    headers=self.headers,
                    json=payload,
                    timeout=timeout
                )
                
                self.stats['total_api_calls'] += 1
                
                if response.status_code == 200:
                    result = response.json()
                    return result
                
                elif response.status_code == 503:  # Model loading
                    wait_time = min(15 + attempt * 10, 60)
                    if attempt < max_retries:
                        st.info(f"🔄 Model loading... retrying in {wait_time}s")
                        time.sleep(wait_time)
                        continue
                    else:
                        return {"error": f"Model loading timeout: {model}"}
                
                elif response.status_code == 429:  # Rate limit
                    self.stats['rate_limit_hits'] += 1
                    wait_time = min(30 + attempt * 15, 120)
                    if attempt < max_retries:
                        st.warning(f"⏰ Rate limited, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return {"error": "Rate limit exceeded"}
                
                else:
                    error_text = response.text
                    return {"error": f"API error {response.status_code}: {error_text}"}
                
            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    st.warning(f"⏰ Timeout, retrying... ({attempt + 1}/{max_retries + 1})")
                    time.sleep(5)
                    continue
                else:
                    return {"error": "Request timeout"}
            
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(2)
                    continue
                else:
                    self.stats['errors'] += 1
                    return {"error": str(e)}
        
        return {"error": "Max retries exceeded"}
    
    def _get_cache_key(self, content: str, task_type: str, model: str) -> str:
        """Generate cache key"""
        cache_content = f"{task_type}:{model}:{content}"
        return hashlib.md5(cache_content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[str]:
        """Load response from cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    # Check cache age (2 hours for HF free tier)
                    cache_time = datetime.fromisoformat(cached_data.get('timestamp', ''))
                    if (datetime.now() - cache_time).total_seconds() < 7200:
                        self.stats['cache_hits'] += 1
                        return cached_data.get('response', '')
            except:
                pass
        return None
    
    def _save_to_cache(self, cache_key: str, response: str, task_type: str, model: str):
        """Save response to cache"""
        try:
            cache_data = {
                'response': response,
                'timestamp': datetime.now().isoformat(),
                'task_type': task_type,
                'model': model
            }
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except:
            pass
    
    def extract_resume_data(self, resume_text: str) -> Dict[str, Any]:
        """Extract structured data from resume using HuggingFace LLM"""
        result = {
            'success': False,
            'data': {},
            'error': None,
            'processing_time': 0
        }
        
        if not self.api_token or not self.connection_status.get('connected', False):
            result['error'] = "HuggingFace API not available"
            return result
        
        start_time = time.time()
        
        try:
            from config import get_huggingface_prompt, optimize_prompt_for_free_tier
            
            # Create optimized prompt for data extraction
            prompt = get_huggingface_prompt(
                "data_extraction",
                resume_text=resume_text[:2000]  # Limit for free tier
            )
            
            # Optimize for free tier
            prompt = optimize_prompt_for_free_tier(prompt, 1500)
            
            # Check cache first
            cache_key = self._get_cache_key(prompt, "data_extraction", self.data_extraction_model)
            cached_response = self._load_from_cache(cache_key)
            
            if cached_response:
                result.update({
                    'success': True,
                    'data': self._parse_json_response(cached_response),
                    'processing_time': time.time() - start_time
                })
                return result
            
            # Call HuggingFace API
            with st.spinner("🤗 Extracting data with HuggingFace..."):
                api_response = self._call_llm_api(
                    prompt, 
                    self.data_extraction_model,
                    max_length=self.settings["data_extraction"]["max_length"],
                    temperature=self.settings["data_extraction"]["temperature"],
                    timeout=self.settings["data_extraction"]["timeout"]
                )
                
                if api_response and "error" not in api_response:
                    # Extract generated text
                    if isinstance(api_response, list) and len(api_response) > 0:
                        generated_text = api_response[0].get('generated_text', '')
                    else:
                        generated_text = str(api_response)
                    
                    if generated_text:
                        # Save to cache
                        self._save_to_cache(cache_key, generated_text, "data_extraction", self.data_extraction_model)
                        
                        # Parse structured data
                        extracted_data = self._parse_json_response(generated_text)
                        
                        result.update({
                            'success': True,
                            'data': extracted_data,
                            'raw_response': generated_text
                        })
                        
                        self.stats['data_extractions'] += 1
                    else:
                        result['error'] = "Empty response from HuggingFace"
                else:
                    result['error'] = api_response.get('error', 'API call failed')
        
        except Exception as e:
            result['error'] = f"Data extraction failed: {str(e)}"
            self.stats['errors'] += 1
        
        result['processing_time'] = time.time() - start_time
        return result
    
    def score_relevance(self, job_description: str, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Score candidate relevance using HuggingFace LLM"""
        result = {
            'success': False,
            'score': 0.0,
            'analysis': {},
            'error': None,
            'processing_time': 0
        }
        
        if not self.api_token or not self.connection_status.get('connected', False):
            result['error'] = "HuggingFace API not available"
            return result
        
        start_time = time.time()
        
        try:
            from config import get_huggingface_prompt, optimize_prompt_for_free_tier
            
            # Create scoring prompt
            candidate_summary = self._format_candidate_for_scoring(candidate_data)
            prompt = get_huggingface_prompt(
                "relevance_scoring",
                job_description=job_description[:1200],
                candidate_data=candidate_summary
            )
            
            # Optimize for free tier
            prompt = optimize_prompt_for_free_tier(prompt, 1000)
            
            # Check cache
            cache_key = self._get_cache_key(prompt, "relevance_scoring", self.relevance_scoring_model)
            cached_response = self._load_from_cache(cache_key)
            
            if cached_response:
                parsed = self._parse_scoring_response(cached_response)
                result.update({
                    'success': True,
                    'score': parsed.get('score', 0.0),
                    'analysis': parsed.get('analysis', {}),
                    'processing_time': time.time() - start_time
                })
                return result
            
            # Call HuggingFace API
            with st.spinner("🎯 Scoring relevance with HuggingFace..."):
                api_response = self._call_llm_api(
                    prompt,
                    self.relevance_scoring_model,
                    max_length=self.settings["relevance_scoring"]["max_length"],
                    temperature=self.settings["relevance_scoring"]["temperature"],
                    timeout=self.settings["relevance_scoring"]["timeout"]
                )
                
                if api_response and "error" not in api_response:
                    # Extract generated text
                    if isinstance(api_response, list) and len(api_response) > 0:
                        generated_text = api_response[0].get('generated_text', '')
                    else:
                        generated_text = str(api_response)
                    
                    if generated_text:
                        # Save to cache
                        self._save_to_cache(cache_key, generated_text, "relevance_scoring", self.relevance_scoring_model)
                        
                        # Parse scoring result
                        parsed = self._parse_scoring_response(generated_text)
                        
                        result.update({
                            'success': True,
                            'score': parsed.get('score', 0.0),
                            'analysis': parsed.get('analysis', {}),
                            'raw_response': generated_text
                        })
                        
                        self.stats['relevance_scores'] += 1
                    else:
                        result['error'] = "Empty response from HuggingFace"
                else:
                    result['error'] = api_response.get('error', 'API call failed')
        
        except Exception as e:
            result['error'] = f"Relevance scoring failed: {str(e)}"
            self.stats['errors'] += 1
        
        result['processing_time'] = time.time() - start_time
        return result
    
    def generate_detailed_analysis(self, job_title: str, job_description: str, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed analysis using HuggingFace LLM"""
        result = {
            'success': False,
            'analysis': '',
            'error': None,
            'processing_time': 0
        }
        
        if not self.api_token or not self.connection_status.get('connected', False):
            result['error'] = "HuggingFace API not available"
            return result
        
        start_time = time.time()
        
        try:
            from config import get_huggingface_prompt, optimize_prompt_for_free_tier
            
            # Create detailed analysis prompt
            candidate_summary = self._format_candidate_for_analysis(candidate_data)
            prompt = get_huggingface_prompt(
                "detailed_analysis",
                job_title=job_title,
                job_description=job_description[:1200],
                candidate_data=candidate_summary
            )
            
            # Optimize for free tier
            prompt = optimize_prompt_for_free_tier(prompt, 1000)
            
            # Check cache
            cache_key = self._get_cache_key(prompt, "detailed_analysis", self.main_model)
            cached_response = self._load_from_cache(cache_key)
            
            if cached_response:
                result.update({
                    'success': True,
                    'analysis': cached_response,
                    'processing_time': time.time() - start_time
                })
                return result
            
            # Call HuggingFace API
            with st.spinner("📊 Generating analysis with HuggingFace..."):
                api_response = self._call_llm_api(
                    prompt,
                    self.main_model,
                    max_length=800,  # Longer for detailed analysis
                    temperature=0.4,  # More creative for analysis
                    timeout=60
                )
                
                if api_response and "error" not in api_response:
                    # Extract generated text
                    if isinstance(api_response, list) and len(api_response) > 0:
                        generated_text = api_response[0].get('generated_text', '')
                    else:
                        generated_text = str(api_response)
                    
                    if generated_text:
                        # Save to cache
                        self._save_to_cache(cache_key, generated_text, "detailed_analysis", self.main_model)
                        
                        result.update({
                            'success': True,
                            'analysis': generated_text.strip()
                        })
                        
                        self.stats['detailed_analyses'] += 1
                    else:
                        result['error'] = "Empty response from HuggingFace"
                else:
                    result['error'] = api_response.get('error', 'API call failed')
        
        except Exception as e:
            result['error'] = f"Analysis generation failed: {str(e)}"
            self.stats['errors'] += 1
        
        result['processing_time'] = time.time() - start_time
        return result
    
    def chat_response(self, question: str, context: str = None) -> Dict[str, Any]:
        """Generate chatbot response using HuggingFace LLM"""
        result = {
            'success': False,
            'answer': '',
            'error': None,
            'processing_time': 0
        }
        
        if not self.api_token or not self.connection_status.get('connected', False):
            result['error'] = "HuggingFace API not available"
            return result
        
        start_time = time.time()
        
        try:
            from config import get_huggingface_prompt, optimize_prompt_for_free_tier
            
            # Prepare context-aware prompt
            context_text = context if context else "General HR and resume screening guidance"
            
            prompt = get_huggingface_prompt(
                "chatbot",
                context=context_text[:500],  # Limit context
                question=question
            )
            
            # Optimize for free tier
            prompt = optimize_prompt_for_free_tier(prompt, 800)
            
            # Check cache
            cache_key = self._get_cache_key(prompt, "chatbot", self.chatbot_model)
            cached_response = self._load_from_cache(cache_key)
            
            if cached_response:
                result.update({
                    'success': True,
                    'answer': cached_response,
                    'processing_time': time.time() - start_time
                })
                return result
            
            # Call HuggingFace API
            with st.spinner("💬 HuggingFace thinking..."):
                api_response = self._call_llm_api(
                    prompt,
                    self.chatbot_model,
                    max_length=self.settings["chatbot"]["max_length"],
                    temperature=self.settings["chatbot"]["temperature"],
                    timeout=self.settings["chatbot"]["timeout"]
                )
                
                if api_response and "error" not in api_response:
                    # Extract generated text
                    if isinstance(api_response, list) and len(api_response) > 0:
                        generated_text = api_response[0].get('generated_text', '')
                    else:
                        generated_text = str(api_response)
                    
                    if generated_text:
                        # Clean and format response
                        answer = self._clean_chat_response(generated_text)
                        
                        # Save to cache
                        self._save_to_cache(cache_key, answer, "chatbot", self.chatbot_model)
                        
                        result.update({
                            'success': True,
                            'answer': answer
                        })
                        
                        self.stats['chat_responses'] += 1
                    else:
                        result['error'] = "Empty response from HuggingFace"
                else:
                    result['error'] = api_response.get('error', 'API call failed')
        
        except Exception as e:
            result['error'] = f"Chat error: {str(e)}"
            self.stats['errors'] += 1
        
        result['processing_time'] = time.time() - start_time
        return result
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from HuggingFace response"""
        try:
            # Try to find JSON in response
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            json_matches = re.findall(json_pattern, response)
            
            if json_matches:
                # Try each potential JSON match
                for json_str in json_matches:
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        continue
            
            # If no JSON found, create structured data from text
            return self._extract_structured_data_from_text(response)
        
        except Exception as e:
            return {
                'error': f'Response parse error: {e}',
                'raw': response,
                'name': 'Unknown',
                'skills': [],
                'experience_years': 0
            }
    
    def _extract_structured_data_from_text(self, text: str) -> Dict[str, Any]:
        """Extract structured data from unstructured text"""
        data = {
            'name': 'Unknown',
            'email': '',
            'phone': '',
            'experience_years': 0,
            'current_role': '',
            'skills': [],
            'education': '',
            'companies': [],
            'summary': text[:200] + '...' if len(text) > 200 else text
        }
        
        try:
            # Extract name (simple heuristics)
            name_patterns = [
                r'[Nn]ame[:\s]*([A-Z][a-z]+ [A-Z][a-z]+)',
                r'([A-Z][a-z]+ [A-Z][a-z]+)'
            ]
            for pattern in name_patterns:
                match = re.search(pattern, text)
                if match:
                    data['name'] = match.group(1).strip()
                    break
            
            # Extract email
            email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
            if email_match:
                data['email'] = email_match.group()
            
            # Extract years of experience
            exp_patterns = [
                r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
                r'experience[:\s]*(\d+)\s*(?:years?|yrs?)',
                r'(\d+)\+?\s*(?:years?|yrs?)'
            ]
            for pattern in exp_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    data['experience_years'] = int(match.group(1))
                    break
            
            # Extract skills (basic)
            skill_keywords = ['python', 'java', 'javascript', 'sql', 'react', 'node', 'aws', 'azure']
            found_skills = []
            for skill in skill_keywords:
                if skill.lower() in text.lower():
                    found_skills.append(skill.title())
            data['skills'] = found_skills[:10]  # Limit to 10 skills
            
        except Exception:
            pass  # Return basic data if extraction fails
        
        return data
    
    def _parse_scoring_response(self, response: str) -> Dict[str, Any]:
        """Parse scoring response from HuggingFace"""
        result = {
            'score': 0.0,
            'analysis': {}
        }
        
        try:
            # Look for score patterns
            score_patterns = [
                r'[Ss]core[:\s]*(\d+)',
                r'(\d+)(?:\s*/\s*100)?(?:\s*points?)?',
                r'(\d+)%'
            ]
            
            for pattern in score_patterns:
                match = re.search(pattern, response)
                if match:
                    score = float(match.group(1))
                    # Normalize to 0-1 scale
                    if score > 1:
                        score = score / 100.0
                    result['score'] = max(0.0, min(1.0, score))
                    break
            
            # Extract reasoning
            reason_match = re.search(r'[Rr]eason[:\s]*(.+?)(?:\n|$)', response)
            if reason_match:
                result['analysis']['reasoning'] = reason_match.group(1).strip()
            else:
                result['analysis']['reasoning'] = response[:200] + '...' if len(response) > 200 else response
            
        except Exception:
            # Fallback scoring based on keywords
            response_lower = response.lower()
            if any(word in response_lower for word in ['excellent', 'outstanding', 'perfect']):
                result['score'] = 0.9
            elif any(word in response_lower for word in ['good', 'strong', 'solid']):
                result['score'] = 0.75
            elif any(word in response_lower for word in ['fair', 'adequate', 'decent']):
                result['score'] = 0.6
            elif any(word in response_lower for word in ['poor', 'weak', 'lacking']):
                result['score'] = 0.3
            else:
                result['score'] = 0.5
            
            result['analysis']['reasoning'] = "Score based on qualitative assessment"
        
        return result
    
    def _clean_chat_response(self, response: str) -> str:
        """Clean and format chat response"""
        # Remove common artifacts
        cleaned = response.strip()
        
        # Remove instruction artifacts
        cleaned = re.sub(r'\[INST\].*?\[/INST\]', '', cleaned)
        cleaned = re.sub(r'<s>|</s>', '', cleaned)
        
        # Limit length for chat
        if len(cleaned) > 500:
            sentences = cleaned.split('. ')
            truncated = []
            current_length = 0
            
            for sentence in sentences:
                if current_length + len(sentence) < 450:
                    truncated.append(sentence)
                    current_length += len(sentence)
                else:
                    break
            
            cleaned = '. '.join(truncated)
            if not cleaned.endswith('.'):
                cleaned += '.'
        
        return cleaned.strip()
    
    def _format_candidate_for_scoring(self, candidate_data: Dict[str, Any]) -> str:
        """Format candidate data for scoring prompt"""
        data = candidate_data.get('data', candidate_data)
        
        formatted = f"""
Name: {data.get('name', 'Unknown')}
Experience: {data.get('experience_years', 0)} years
Current Role: {data.get('current_role', 'Not specified')}
Skills: {', '.join(data.get('skills', [])[:8])}
Education: {data.get('education', 'Not specified')}
""".strip()
        
        return formatted
    
    def _format_candidate_for_analysis(self, candidate_data: Dict[str, Any]) -> str:
        """Format candidate data for detailed analysis"""
        data = candidate_data.get('data', candidate_data)
        
        formatted = f"""
Name: {data.get('name', 'Unknown')}
Experience: {data.get('experience_years', 0)} years
Current Role: {data.get('current_role', 'Not specified')}
Skills: {', '.join(data.get('skills', []))}
Education: {data.get('education', 'Not specified')}
Companies: {', '.join(data.get('companies', [])[:3])}
Summary: {data.get('summary', 'Not provided')[:300]}
""".strip()
        
        return formatted
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.json')]
            total_size = sum(
                os.path.getsize(os.path.join(self.cache_dir, f))
                for f in cache_files
            )
            
            stats = self.stats.copy()
            stats.update({
                'cache_enabled': True,
                'cache_files': len(cache_files),
                'cache_size_mb': round(total_size / (1024 * 1024), 2),
                'cache_hit_rate': (
                    stats['cache_hits'] / stats['total_api_calls']
                    if stats['total_api_calls'] > 0 else 0
                ),
                'error_rate': (
                    stats['errors'] / stats['total_api_calls']
                    if stats['total_api_calls'] > 0 else 0
                ),
                'rate_limit_percentage': (
                    stats['rate_limit_hits'] / stats['total_api_calls'] * 100
                    if stats['total_api_calls'] > 0 else 0
                )
            })
            
            return stats
            
        except Exception:
            return self.stats.copy()
    
    def clear_cache(self):
        """Clear HuggingFace cache"""
        try:
            import shutil
            cache_files = len([f for f in os.listdir(self.cache_dir) if f.endswith('.json')])
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            st.success(f"✅ Cleared {cache_files} HuggingFace cache entries")
        except Exception as e:
            st.error(f"❌ Failed to clear cache: {e}")
    
    def run_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive diagnostics"""
        diagnostics = {
            'timestamp': datetime.now().isoformat(),
            'connection_test': self.connection_status,
            'model_tests': {},
            'performance_stats': self.get_cache_stats(),
            'rate_limit_status': f"{self.stats['rate_limit_hits']} rate limit hits"
        }
        
        # Test different models
        test_models = [self.data_extraction_model, self.chatbot_model]
        
        for model in test_models[:2]:  # Limit to 2 to avoid rate limits
            try:
                test_result = self._call_llm_api("Test", model, max_length=20, timeout=10)
                if test_result and "error" not in test_result:
                    diagnostics['model_tests'][model.split('/')[-1]] = {'status': 'available'}
                else:
                    diagnostics['model_tests'][model.split('/')[-1]] = {
                        'status': 'failed', 
                        'error': test_result.get('error', 'Unknown')
                    }
            except Exception as e:
                diagnostics['model_tests'][model.split('/')[-1]] = {'status': 'error', 'error': str(e)}
        
        return diagnostics