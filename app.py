import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

# Configure page
st.set_page_config(
    page_title="Resume Screening System - HuggingFace Only",
    page_icon="🤗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'screening_engine' not in st.session_state:
    st.session_state.screening_engine = None
if 'last_results' not in st.session_state:
    st.session_state.last_results = None
if 'chatbot_history' not in st.session_state:
    st.session_state.chatbot_history = []

def main():
    """Main application function"""
    
    # Header
    st.title("🤗 AI Resume Screening System")
    st.markdown("### **HuggingFace Only:** Free Inference API for All Tasks")
    
    # Initialize screening engine with error handling
    if st.session_state.screening_engine is None:
        with st.spinner("🚀 Initializing HuggingFace system..."):
            try:
                from models.screening_engine import HuggingFaceScreeningEngine
                st.session_state.screening_engine = HuggingFaceScreeningEngine()
                st.success("✅ HuggingFace system initialized!")
            except Exception as e:
                st.error(f"❌ System initialization failed: {e}")
                st.info("💡 Check your HuggingFace API token in secrets.toml")
                return
    
    # Sidebar - System Status
    render_sidebar()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Resume Screening", 
        "📊 Results & Analysis", 
        "💬 AI Chatbot", 
        "⚙️ System Status"
    ])
    
    with tab1:
        render_resume_screening()
    
    with tab2:
        render_results_analysis()
    
    with tab3:
        render_ai_chatbot()
    
    with tab4:
        render_system_status()

def render_sidebar():
    """Render sidebar with system info"""
    st.sidebar.title("🤗 HuggingFace System")
    
    # Quick status check
    if st.session_state.screening_engine:
        try:
            system_status = st.session_state.screening_engine.get_system_status()
            
            if system_status.get('ready_for_screening', False):
                st.sidebar.success("✅ HuggingFace Ready")
            else:
                st.sidebar.error("❌ HuggingFace Issues")
                
                # Show what's broken
                hf_status = system_status.get('huggingface_status', {})
                conn_status = hf_status.get('connection_status', {})
                
                if not conn_status.get('connected', False):
                    st.sidebar.write("❌ API connection failed")
                    error = conn_status.get('error', 'Unknown error')
                    st.sidebar.write(f"Error: {error}")
            
            # Model info with FREE emphasis
            st.sidebar.info("""
            **🤗 All Tasks:** Mistral 7B (FREE!)
            
            **✅ Data Extraction**
            **✅ Relevance Scoring** 
            **✅ Detailed Analysis**
            **✅ Chatbot**
            
            **💰 Cost:** FREE with rate limits
            """)
            
            # Stats
            stats = system_status.get('processing_stats', {})
            if stats.get('total_resumes_processed', 0) > 0:
                st.sidebar.metric("Resumes Processed", stats['total_resumes_processed'])
                st.sidebar.metric("HuggingFace Calls", stats['huggingface_calls'])
                st.sidebar.metric("Rate Limit Hits", stats.get('rate_limit_hits', 0))
                
        except Exception as e:
            st.sidebar.error("❌ System Error")
            st.sidebar.write(f"Error: {e}")
    else:
        st.sidebar.warning("⚠️ System not initialized")
    
    # Navigation
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📋 Quick Actions**")
    
    if st.sidebar.button("🔄 Refresh System"):
        st.session_state.screening_engine = None
        st.rerun()
    
    if st.sidebar.button("🧹 Clear Cache"):
        if st.session_state.screening_engine:
            try:
                st.session_state.screening_engine.clear_cache()
                st.sidebar.success("✅ Cache cleared")
            except Exception as e:
                st.sidebar.error(f"❌ Cache clear failed: {e}")
    
    # Rate limit info
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **⏰ Rate Limits (Free Tier)**
    
    ~10 requests/minute per model
    
    💡 **Tips:**
    - Process in small batches
    - Be patient with processing
    - Upgrade to Pro for faster speeds
    """)

def render_resume_screening():
    """Main resume screening interface"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📄 Upload Resumes")
        
        # Check system status first
        if not st.session_state.screening_engine:
            st.error("❌ System not initialized. Please refresh.")
            return
        
        try:
            system_status = st.session_state.screening_engine.get_system_status()
            
            if not system_status.get('ready_for_screening', False):
                st.warning("⚠️ HuggingFace not ready. Check API token in System Status.")
                return
                
        except Exception as e:
            st.error(f"❌ System check failed: {e}")
            return
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Select resume files",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'doc', 'txt'],
            help="Upload PDF, Word documents, or text files"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
            
            # Rate limit warning for large batches
            if len(uploaded_files) > 10:
                st.warning("⚠️ Large batch detected. Processing may be slower due to free tier rate limits.")
                st.info("💡 Consider processing in smaller batches or upgrading to HuggingFace Pro")
            
            # Show file details
            with st.expander("📁 File Details", expanded=False):
                for file in uploaded_files:
                    st.write(f"• {file.name} ({file.size/1024:.1f} KB)")
    
    with col2:
        st.header("💼 Job Description")
        
        # Job templates
        from config import JOB_TEMPLATES
        
        template_choice = st.selectbox("Choose template:", ["Custom"] + list(JOB_TEMPLATES.keys()))
        
        if template_choice != "Custom":
            job_description = st.text_area(
                "Job Description",
                value=JOB_TEMPLATES[template_choice],
                height=200,
                help="Edit the template or write your own (shorter is better for free tier)"
            )
        else:
            job_description = st.text_area(
                "Job Description",
                height=200,
                placeholder="Paste job description here...",
                help="Keep concise for better free tier performance"
            )
    
    # Processing section
    st.markdown("---")
    
    if uploaded_files and job_description:
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🤗 Analyze with HuggingFace", type="primary", use_container_width=True):
                process_resumes(uploaded_files, job_description)
        
        with col2:
            if st.button("📊 Quick Preview", use_container_width=True):
                show_quick_preview(uploaded_files, job_description)
        
        with col3:
            if st.button("💰 Cost Info (FREE!)", use_container_width=True):
                show_cost_info(len(uploaded_files))
    
    elif uploaded_files and not job_description:
        st.warning("⚠️ Please provide a job description to analyze resumes")
    elif not uploaded_files:
        st.info("💡 Upload resume files to get started")

def process_resumes(uploaded_files, job_description):
    """Process uploaded resumes with HuggingFace"""
    
    st.markdown("### 🤗 Processing with HuggingFace Free API...")
    
    # Show rate limiting info
    if len(uploaded_files) > 5:
        st.info("⏰ Free tier processing: ~1-2 minutes per resume due to rate limits")
    
    start_time = datetime.now()
    
    # Process with HuggingFace engine
    results = st.session_state.screening_engine.process_multiple_resumes(
        uploaded_files, job_description
    )
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # Store results
    st.session_state.last_results = results
    
    # Show summary
    summary = results.get('summary', {})
    
    st.success(f"✅ HuggingFace processing completed in {processing_time:.1f} seconds!")
    
    # Quick metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Processed", summary.get('total_processed', 0))
    
    with col2:
        st.metric("Successful", summary.get('successful', 0))
    
    with col3:
        st.metric("Average Score", f"{summary.get('avg_composite_score', 0):.3f}")
    
    with col4:
        hf_calls = summary.get('huggingface_calls', 0)
        st.metric("HF API Calls (FREE!)", hf_calls)
    
    # Show top candidates
    if results.get('results'):
        st.markdown("### 🏆 Top Candidates (HuggingFace Analysis)")
        
        top_candidates = results['results'][:5]  # Top 5
        
        for i, candidate in enumerate(top_candidates):
            with st.expander(f"#{i+1} {candidate['filename']} - Score: {candidate['scores']['composite_score']:.3f}"):
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 HuggingFace Scores**")
                    scores = candidate['scores']
                    st.write(f"• Relevance (Mistral): {scores['relevance_score']:.3f}")
                    st.write(f"• Experience: {scores['experience_score']:.3f}")
                    st.write(f"• Skills Match: {scores['skills_score']:.3f}")
                
                with col2:
                    st.markdown("**🤗 HuggingFace Insights**")
                    analysis = candidate.get('analysis', {})
                    
                    # Show candidate summary
                    summary = analysis.get('candidate_summary', 'N/A')
                    st.write(f"**Summary:** {summary}")
                    
                    # Show key highlights
                    highlights = analysis.get('key_highlights', [])
                    if highlights:
                        for highlight in highlights[:3]:
                            st.write(f"• {highlight}")
                
                # Show detailed analysis if available
                detailed = analysis.get('detailed_analysis', '')
                if detailed:
                    st.markdown("**🎯 Detailed Mistral Analysis:**")
                    st.write(detailed[:300] + "..." if len(detailed) > 300 else detailed)
                
                # Show models used
                models_used = analysis.get('models_used', {})
                if models_used:
                    st.caption(f"Models: {models_used.get('data_extraction', 'Mistral 7B')} | Cost: {models_used.get('cost', 'FREE')}")
        
        # Export option
        st.markdown("---")
        if st.button("📥 Export HuggingFace Results"):
            csv_path = st.session_state.screening_engine.export_results_to_csv()
            if csv_path:
                st.success(f"✅ Results exported to {csv_path}")
                
                # Offer download
                with open(csv_path, 'rb') as f:
                    st.download_button(
                        "⬇️ Download CSV",
                        f.read(),
                        file_name=f"huggingface_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

def show_quick_preview(uploaded_files, job_description):
    """Show quick preview"""
    st.info("🔍 Quick Preview - HuggingFace Free Processing")
    
    # Show processing details
    st.write(f"**Files to process:** {len(uploaded_files)}")
    st.write(f"**Job description length:** {len(job_description)} characters")
    
    # Estimated processing time (accounting for rate limits)
    base_time = len(uploaded_files) * 8  # Base processing time
    rate_limit_time = max(0, (len(uploaded_files) - 5) * 6)  # Additional time for rate limits
    estimated_time = base_time + rate_limit_time
    st.write(f"**Estimated processing time:** {estimated_time} seconds")
    
    # Show processing steps
    st.write("**HuggingFace Processing Steps:**")
    st.write("1. 📄 Parse resume text")
    st.write("2. 🤗 Extract structured data with Mistral 7B")
    st.write("3. 🎯 Score relevance with Mistral 7B")
    st.write("4. 📊 Generate detailed analysis with Mistral 7B")
    st.write("5. 💬 Prepare chatbot context")
    
    # Rate limiting info
    if len(uploaded_files) > 10:
        st.warning("⚠️ Large batch - expect slower processing due to free tier rate limits")
        st.info("💡 Tip: Process in batches of 5-10 for optimal speed")
    
    # Show files
    st.markdown("**Files:**")
    for i, file in enumerate(uploaded_files[:3]):
        st.write(f"• {file.name}")
    
    if len(uploaded_files) > 3:
        st.write(f"• ... and {len(uploaded_files) - 3} more files")

def show_cost_info(num_files):
    """Show HuggingFace cost information (FREE!)"""
    if st.session_state.screening_engine:
        cost_info = st.session_state.screening_engine.get_cost_estimates(num_files)
        
        st.success("💰 HuggingFace Free Tier - No Cost!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Cost", "$0.00 (FREE!)")
            st.metric("Cost per Resume", "$0.00 (FREE!)")
            
            rate_info = cost_info['rate_limit_info']
            st.metric("Est. Processing Time", f"{rate_info['estimated_processing_time_minutes']:.1f} min")
        
        with col2:
            breakdown = cost_info['calls_breakdown']
            st.write("**API Calls (All FREE):**")
            st.write(f"• Data extraction: {breakdown['data_extraction']}")
            st.write(f"• Relevance scoring: {breakdown['relevance_scoring']}")
            st.write(f"• Detailed analysis: {breakdown['detailed_analysis']}")
            st.write(f"• **Total calls: {breakdown['total_calls']}**")
        
        # Upgrade info
        if 'upgrade_option' in cost_info:
            upgrade = cost_info['upgrade_option']
            st.info(f"💡 **Upgrade Option:** {upgrade['huggingface_pro']} - {upgrade['benefits']}")
        
        st.success("🎉 " + cost_info['note'])

def render_results_analysis():
    """Render detailed results analysis"""
    
    if not st.session_state.last_results:
        st.info("📊 No results to display. Please process some resumes first.")
        return
    
    results = st.session_state.last_results
    summary = results.get('summary', {})
    
    # Summary metrics
    st.header("📊 HuggingFace Analysis Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Resumes", summary.get('total_processed', 0))
    
    with col2:
        success_rate = (summary.get('successful', 0) / max(summary.get('total_processed', 1), 1) * 100)
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with col3:
        st.metric("Avg Score", f"{summary.get('avg_composite_score', 0):.3f}")
    
    with col4:
        st.metric("HF API Calls", summary.get('huggingface_calls', 0))
    
    # HuggingFace Performance
    if 'huggingface_performance' in summary:
        st.markdown("### 🤗 HuggingFace Performance")
        
        perf = summary['huggingface_performance']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Successful Extractions", perf.get('successful_extractions', 0))
            st.metric("Successful Scores", perf.get('successful_scores', 0))
        
        with col2:
            st.metric("Rate Limit Hits", perf.get('rate_limit_hits', 0))
            st.metric("Avg Processing Time", f"{perf.get('avg_processing_time', 0):.1f}s")
        
        with col3:
            st.metric("Error Rate", f"{perf.get('error_rate', 0)*100:.1f}%")
            st.metric("Total Cost", perf.get('cost', 'FREE'))
    
    # Score distribution
    if 'score_distribution' in summary:
        st.markdown("### 📈 Score Distribution")
        
        dist = summary['score_distribution']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mean", f"{dist.get('mean', 0):.3f}")
            st.metric("Median", f"{dist.get('median', 0):.3f}")
        
        with col2:
            st.metric("Min Score", f"{dist.get('min', 0):.3f}")
            st.metric("Max Score", f"{dist.get('max', 0):.3f}")
        
        with col3:
            st.metric("Std Dev", f"{dist.get('std', 0):.3f}")
            quartiles = dist.get('quartiles', {})
            st.metric("Q1-Q3", f"{quartiles.get('q1', 0):.2f} - {quartiles.get('q3', 0):.2f}")
    
    # HuggingFace insights
    if 'huggingface_insights' in results:
        st.markdown("### 🧠 HuggingFace Quality Insights")
        
        insights = results['huggingface_insights']
        
        if 'huggingface_performance' in insights:
            perf_insights = insights['huggingface_performance']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"""
                **🤗 Mistral Performance**
                
                Extraction Success: {perf_insights.get('extraction_success_rate', 0)*100:.1f}%
                
                Analysis Success: {perf_insights.get('analysis_success_rate', 0)*100:.1f}%
                
                Reliability: {perf_insights.get('overall_reliability', 'Unknown')}
                
                Cost: {perf_insights.get('cost_analysis', 'FREE')}
                """)
            
            with col2:
                rate_limiting = insights.get('rate_limiting', {})
                st.info(f"""
                **⏰ Rate Limiting**
                
                Hits Detected: {rate_limiting.get('hits_detected', 0)}
                
                Batch Size: {rate_limiting.get('recommended_batch_size', '5-10')}
                
                Strategy: {rate_limiting.get('processing_strategy', 'Small batches')}
                
                Upgrade: {rate_limiting.get('upgrade_suggestion', 'Consider Pro')}
                """)
    
    # Detailed results table
    st.markdown("### 📋 Detailed HuggingFace Results")
    
    if results.get('results'):
        # Create results DataFrame
        result_data = []
        for result in results['results']:
            scores = result.get('scores', {})
            analysis = result.get('analysis', {})
            extracted_data = analysis.get('extracted_data', {})
            
            result_data.append({
                'Rank': result.get('rank', ''),
                'Filename': result['filename'],
                'Candidate': extracted_data.get('name', 'Unknown'),
                'Composite Score': scores.get('composite_score', 0),
                'Relevance (Mistral)': scores.get('relevance_score', 0),
                'Experience': scores.get('experience_score', 0),
                'Skills': scores.get('skills_score', 0),
                'Years Exp': extracted_data.get('experience_years', 0),
                'HF Calls': result.get('huggingface_calls', 0),
                'Analysis': 'Yes' if analysis.get('detailed_analysis') else 'No',
                'Cost': 'FREE'
            })
        
        df = pd.DataFrame(result_data)
        
        # Display with formatting
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "Composite Score": st.column_config.ProgressColumn(
                    "Composite Score",
                    help="Overall candidate score",
                    min_value=0,
                    max_value=1,
                ),
                "Relevance (Mistral)": st.column_config.ProgressColumn(
                    "Relevance (Mistral)",
                    help="Mistral 7B relevance score",
                    min_value=0,
                    max_value=1,
                )
            }
        )

def render_ai_chatbot():
    """Render HuggingFace chatbot interface"""
    
    st.header("💬 HuggingFace Resume Chatbot")
    st.markdown("**Powered by Mistral 7B Instruct (FREE!)**")
    
    if not st.session_state.screening_engine:
        st.error("❌ System not initialized")
        return
    
    # Check if HuggingFace is ready
    try:
        system_status = st.session_state.screening_engine.get_system_status()
        hf_ready = system_status.get('ready_for_screening', False)
        
        if not hf_ready:
            st.warning("⚠️ HuggingFace not connected. Check your API token.")
            return
            
    except Exception as e:
        st.error(f"❌ System check failed: {e}")
        return
    
    # Resume context selection
    has_results = bool(st.session_state.last_results)
    
    if has_results:
        results = st.session_state.last_results['results']
        resume_options = ["General Questions"] + [r['filename'] for r in results[:10]]
        selected_resume = st.selectbox("Choose resume context:", resume_options)
        
        if selected_resume != "General Questions":
            # Find selected resume data
            resume_data = next((r for r in results if r['filename'] == selected_resume), None)
            if resume_data:
                st.success(f"✅ Using {selected_resume} as context for Mistral")
                
                # Show brief resume info
                with st.expander("📋 Resume Context for Mistral"):
                    analysis = resume_data.get('analysis', {})
                    extracted_data = analysis.get('extracted_data', {})
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Candidate:**")
                        st.write(f"• Name: {extracted_data.get('name', 'Unknown')}")
                        st.write(f"• Experience: {extracted_data.get('experience_years', 0)} years")
                    
                    with col2:
                        st.write("**Mistral Analysis:**")
                        st.write(f"• Composite Score: {resume_data['scores']['composite_score']:.3f}")
                        st.write(f"• Relevance Score: {resume_data['scores']['relevance_score']:.3f}")
    else:
        selected_resume = "General Questions"
        st.info("💡 Process some resumes to get candidate-specific insights from Mistral")
    
    # Chat history
    if st.session_state.chatbot_history:
        st.markdown("### 📜 Chat History with Mistral")
        
        for i, exchange in enumerate(st.session_state.chatbot_history[-5:]):  # Show last 5
            with st.expander(f"Chat {i+1}: {exchange['question'][:50]}..."):
                st.write(f"**Q:** {exchange['question']}")
                st.write(f"**Mistral:** {exchange['answer']}")
                st.caption(f"Time: {exchange.get('time', 'Unknown')} | Cost: FREE")
    
    # Chat input
    st.markdown("### 💬 Ask Mistral (FREE!)")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_question = st.text_input(
            "Ask Mistral about resumes or hiring:",
            placeholder="e.g., What are this candidate's main strengths?",
            key="mistral_chat_input"
        )
    
    with col2:
        ask_button = st.button("💬 Ask Mistral", type="primary")
    
    if ask_button and user_question:
        process_mistral_chat(user_question, selected_resume)
    
    # Suggested questions
    st.markdown("### 💡 Suggested Questions for Mistral")
    
    suggested_questions = get_suggested_questions(selected_resume)
    
    cols = st.columns(2)
    for i, suggestion in enumerate(suggested_questions[:6]):
        col = cols[i % 2]
        with col:
            if st.button(f"💭 {suggestion}", key=f"mistral_suggest_{i}"):
                process_mistral_chat(suggestion, selected_resume)
    
    # Clear history
    if st.button("🧹 Clear Chat History"):
        st.session_state.chatbot_history = []
        st.rerun()
    
    # Rate limit warning
    st.info("⏰ **Free Tier Note:** Chat responses may be slower due to rate limits. Upgrade to HuggingFace Pro for faster responses.")

def process_mistral_chat(question, selected_resume):
    """Process chatbot query with HuggingFace"""
    
    # Get resume context if selected
    resume_context = None
    if selected_resume != "General Questions" and st.session_state.last_results:
        results = st.session_state.last_results['results']
        resume_data = next((r for r in results if r['filename'] == selected_resume), None)
        
        if resume_data:
            # Prepare context from extracted data
            analysis = resume_data.get('analysis', {})
            extracted_data = analysis.get('extracted_data', {})
            
            # Create context string
            context_parts = []
            context_parts.append(f"Candidate: {extracted_data.get('name', 'Unknown')}")
            context_parts.append(f"Experience: {extracted_data.get('experience_years', 0)} years")
            context_parts.append(f"Skills: {', '.join(extracted_data.get('skills', [])[:5])}")
            
            # Add score information
            scores = resume_data.get('scores', {})
            context_parts.append(f"Score: {scores.get('composite_score', 0):.3f}")
            
            resume_context = "; ".join(context_parts)
    
    # Get response from HuggingFace
    try:
        with st.spinner("🤗 Mistral is thinking... (FREE tier may be slower)"):
            hf_manager = st.session_state.screening_engine.huggingface_manager
            response = hf_manager.chat_response(question, resume_context)
        
        if response['success']:
            st.success("✅ Response from Mistral 7B (FREE!):")
            st.write(response['answer'])
            
            # Add to history
            st.session_state.chatbot_history.append({
                'question': question,
                'answer': response['answer'],
                'model': 'Mistral 7B Instruct (FREE)',
                'time': datetime.now().strftime("%H:%M:%S"),
                'resume_context': selected_resume,
                'cost': 'FREE'
            })
            
            # Auto-refresh to show in history
            st.rerun()
        else:
            error_msg = response.get('error', 'Unknown error')
            st.error(f"❌ Mistral chat failed: {error_msg}")
            
            if "rate limit" in error_msg.lower():
                st.info("⏰ **Rate Limited:** Please wait a moment and try again. Consider upgrading to HuggingFace Pro for higher limits.")
            
    except Exception as e:
        st.error(f"❌ Chat error: {e}")

def get_suggested_questions(selected_resume):
    """Get context-appropriate suggested questions"""
    
    if selected_resume == "General Questions":
        return [
            "How do I evaluate technical skills effectively?",
            "What are red flags to look for in resumes?",
            "How to assess cultural fit during screening?",
            "What questions should I ask in interviews?",
            "How to improve our hiring process?",
            "What makes a strong candidate profile?"
        ]
    else:
        return [
            "What are this candidate's main strengths?",
            "How does their experience align with our needs?",
            "What questions should I ask them in an interview?",
            "Are there any concerns about this candidate?",
            "How does this candidate compare to others?",
            "What role would be best for this candidate?"
        ]

def render_system_status():
    """Render system status and diagnostics"""
    
    st.header("⚙️ HuggingFace System Status")
    
    if not st.session_state.screening_engine:
        st.error("❌ System not initialized")
        
        if st.button("🔄 Try Initialize"):
            st.session_state.screening_engine = None
            st.rerun()
        return
    
    # Get system status
    try:
        system_status = st.session_state.screening_engine.get_system_status()
        
        # Overall status
        if system_status.get('ready_for_screening', False):
            st.success("✅ HuggingFace System Ready for Resume Screening (FREE!)")
        else:
            st.error("❌ HuggingFace Issues Detected")
        
        # HuggingFace Status Details
        st.markdown("### 🤗 HuggingFace (Mistral) Status")
        
        hf_status = system_status.get('huggingface_status', {})
        conn_status = hf_status.get('connection_status', {})
        
        if conn_status.get('connected', False):
            st.success("✅ Mistral 7B Connected and Ready (FREE!)")
            
            models = hf_status.get('models', {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"""
                **🤗 Active Models (All FREE)**
                
                Data Extraction: {models.get('data_extraction', 'Mistral 7B')}
                
                Scoring: {models.get('relevance_scoring', 'Mistral 7B')}
                
                Chatbot: {models.get('chatbot', 'Mistral 7B')}
                """)
            
            with col2:
                st.info(f"""
                **💰 Cost Information**
                
                API Calls: FREE
                
                Rate Limits: {hf_status.get('rate_limits', 'Yes')}
                
                Upgrade: HuggingFace Pro ($9/month)
                """)
        else:
            st.error("❌ Mistral Connection Failed")
            error_msg = conn_status.get('error', 'Unknown error')
            st.error(f"**Error:** {error_msg}")
            
            if "rate limit" in error_msg.lower():
                st.info("⏰ **Rate Limited:** Wait and try again, or upgrade to Pro")
        
        # Performance Stats
        st.markdown("### 📊 Performance Statistics")
        
        stats = system_status.get('processing_stats', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Resumes Processed", stats.get('total_resumes_processed', 0))
        
        with col2:
            st.metric("Avg Processing Time", f"{stats.get('avg_processing_time', 0):.2f}s")
        
        with col3:
            st.metric("Total HF Calls", stats.get('huggingface_calls', 0))
        
        with col4:
            st.metric("Rate Limit Hits", stats.get('rate_limit_hits', 0))
        
        # Rate Limiting Analysis
        if stats.get('rate_limit_hits', 0) > 0:
            st.markdown("### ⏰ Rate Limiting Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.warning(f"""
                **Rate Limit Impact**
                
                Total Hits: {stats['rate_limit_hits']}
                
                Success Rate: {((stats.get('huggingface_calls', 0) - stats.get('huggingface_errors', 0)) / max(stats.get('huggingface_calls', 1), 1) * 100):.1f}%
                """)
            
            with col2:
                st.info("""
                **Optimization Tips**
                
                • Process 5-10 resumes at a time
                • Allow 1-2 minutes between batches
                • Upgrade to Pro for unlimited access
                • Use caching to reduce repeat calls
                """)
        
        # Configuration Check
        st.markdown("### 🔧 Configuration")
        
        try:
            from config import HUGGINGFACE_TOKEN
            
            if HUGGINGFACE_TOKEN:
                st.success("✅ HuggingFace API token configured")
                st.info(f"Token length: {len(HUGGINGFACE_TOKEN)} characters")
            else:
                st.error("❌ HuggingFace API token missing")
                st.info("Add HUGGINGFACE_TOKEN to .streamlit/secrets.toml")
                st.info("Get free token from https://huggingface.co/settings/tokens")
                
        except Exception as e:
            st.error(f"Configuration check failed: {e}")
        
        # System Actions
        st.markdown("### 🛠️ System Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🧪 Run Diagnostics"):
                with st.spinner("Running HuggingFace diagnostics..."):
                    diagnostics = st.session_state.screening_engine.run_diagnostics()
                    
                    st.json(diagnostics)
        
        with col2:
            if st.button("🔄 Refresh System"):
                st.session_state.screening_engine = None
                st.rerun()
        
        with col3:
            if st.button("🧹 Clear Cache"):
                try:
                    st.session_state.screening_engine.clear_cache()
                    st.success("✅ Cache cleared")
                except Exception as e:
                    st.error(f"❌ Cache clear failed: {e}")
        
        with col4:
            if st.button("📊 Export Status"):
                status_json = json.dumps(system_status, indent=2, default=str)
                st.download_button(
                    "⬇️ Download JSON",
                    status_json,
                    file_name=f"huggingface_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        # Upgrade Information
        st.markdown("### 🚀 Upgrade Information")
        
        st.info("""
        **HuggingFace Pro Benefits ($9/month):**
        
        • **No rate limits** - Process unlimited resumes
        • **Faster processing** - Priority API access
        • **More models** - Access to premium models
        • **Better support** - Priority customer support
        
        Upgrade at: https://huggingface.co/pricing
        """)
        
    except Exception as e:
        st.error(f"❌ System status check failed: {e}")
        st.write("**Error Details:**", str(e))

if __name__ == "__main__":
    main()