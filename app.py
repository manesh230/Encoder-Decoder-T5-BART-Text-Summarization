import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
import time

# Set page config
st.set_page_config(
    page_title="T5-small LoRA Summarization",
    page_icon="📝",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .summary-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .loading-spinner {
        text-align: center;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📝 T5-small LoRA Text Summarization</h1>', unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading T5 base model and LoRA adapter...")
def load_peft_model():
    """Load the base T5 model and apply LoRA adapter"""
    try:
        # Load the base T5-small model
        base_model_name = "t5-small"
        
        st.write("🔄 Loading base T5-small model...")
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        st.write("🔄 Loading LoRA adapter...")
        # Load the LoRA adapter
        peft_model = PeftModel.from_pretrained(base_model, "manesh1/t5-small-lora-summarization")
        
        # Create pipeline with the combined model
        summarizer = pipeline(
            "summarization",
            model=peft_model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        return summarizer
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_resource(show_spinner="Loading model with alternative method...")
def load_model_alternative():
    """Alternative loading method"""
    try:
        # Try direct pipeline loading with trust_remote_code
        summarizer = pipeline(
            "summarization",
            model="manesh1/t5-small-lora-summarization",
            trust_remote_code=True,
            device=0 if torch.cuda.is_available() else -1
        )
        return summarizer
    except Exception as e:
        st.error(f"Alternative loading failed: {str(e)}")
        return None

def main():
    # Load model
    with st.spinner("🔄 Initializing model... This may take a minute."):
        summarizer = load_peft_model()
        
        # If first method fails, try alternative
        if summarizer is None:
            st.info("Trying alternative loading method...")
            summarizer = load_model_alternative()
    
    if summarizer is None:
        st.error("""
        ❌ Failed to load the model. This is likely because:
        
        - The model is a LoRA adapter and requires special loading
        - PEFT library might not be properly installed
        - Network connectivity issues
        
        **Try these solutions:**
        1. Make sure `peft` is in your requirements.txt
        2. Check your internet connection
        3. Try running locally first
        """)
        return
    
    # Display model status
    device = "GPU 🚀" if torch.cuda.is_available() else "CPU ⚡"
    st.success(f"✅ Model loaded successfully! Running on {device}")
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Settings")
    
    max_length = st.sidebar.slider(
        "Maximum summary length",
        min_value=50,
        max_value=300,
        value=150
    )
    
    min_length = st.sidebar.slider(
        "Minimum summary length", 
        min_value=10,
        max_value=100,
        value=30
    )
    
    # Model info
    st.sidebar.markdown("---")
    st.sidebar.title("ℹ️ Model Information")
    st.sidebar.info("""
    **Architecture:** T5-small + LoRA adapter  
    **Base Model:** t5-small  
    **Adapter:** manesh1/t5-small-lora-summarization  
    **Task:** Text summarization  
    **Framework:** PEFT (Parameter-Efficient Fine-Tuning)
    """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 Input Text")
        
        # Sample texts
        sample_option = st.selectbox(
            "Choose sample text or enter your own:",
            ["Enter your own text", "AI Technology", "Machine Learning", "Climate Change"]
        )
        
        if sample_option == "AI Technology":
            input_text = """Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. Leading AI textbooks define the field as the study of intelligent agents: any system that perceives its environment and takes actions that maximize its chance of achieving its goals. Some popular accounts use the term artificial intelligence to describe machines that mimic cognitive functions that humans associate with the human mind, such as learning and problem solving.

AI applications include advanced web search engines, recommendation systems, understanding human speech, self-driving cars, automated decision-making, and competing at the highest level in strategic game systems. As machines become increasingly capable, tasks considered to require intelligence are often removed from the definition of AI, a phenomenon known as the AI effect."""
        
        elif sample_option == "Machine Learning":
            input_text = """Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data and make predictions or decisions without being explicitly programmed for every task. Deep learning, a further subset of machine learning, uses neural networks with multiple layers to process and extract features from large amounts of data.

These technologies have revolutionized fields like computer vision, natural language processing, and speech recognition. Companies use machine learning for recommendation systems, fraud detection, and automated customer service. The availability of large datasets and powerful computing resources has accelerated advancements in this field."""
        
        elif sample_option == "Climate Change":
            input_text = """Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, but since the 1800s, human activities have been the main driver of climate change, primarily due to the burning of fossil fuels like coal, oil and gas, which produces heat-trapping gases.

The consequences of climate change now include, among others, intense droughts, water scarcity, severe fires, rising sea levels, flooding, melting polar ice, catastrophic storms and declining biodiversity. People are experiencing climate change in diverse ways. It affects our health, ability to grow food, housing, safety and work."""
        
        else:
            input_text = ""
        
        input_text = st.text_area(
            "Your text:",
            value=input_text,
            height=300,
            placeholder="Type or paste your text here...",
            key="input_text"
        )
    
    with col2:
        st.subheader("📋 Summary")
        
        if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
            if not input_text.strip():
                st.warning("⚠️ Please enter some text to summarize.")
            else:
                with st.spinner("⏳ Generating summary..."):
                    try:
                        start_time = time.time()
                        
                        # Generate summary
                        result = summarizer(
                            input_text,
                            max_length=max_length,
                            min_length=min_length,
                            do_sample=False
                        )
                        
                        summary = result[0]['summary_text']
                        end_time = time.time()
                        processing_time = end_time - start_time
                        
                        st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                        st.write(summary)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Statistics
                        col1_stat, col2_stat, col3_stat = st.columns(3)
                        with col1_stat:
                            st.metric("Original Words", len(input_text.split()))
                        with col2_stat:
                            st.metric("Summary Words", len(summary.split()))
                        with col3_stat:
                            reduction = ((len(input_text.split()) - len(summary.split())) / len(input_text.split())) * 100
                            st.metric("Reduction", f"{reduction:.1f}%")
                        
                        st.caption(f"Processing time: {processing_time:.2f} seconds")
                        
                        # Copy to clipboard button
                        st.code(summary, language="text")
                        
                    except Exception as e:
                        st.error(f"❌ Error during summarization: {str(e)}")
        
        else:
            st.info("💡 Enter text on the left and click 'Generate Summary' to see the result here.")
    
    # Technical details
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        **How this model works:**
        
        This is a **Parameter-Efficient Fine-Tuning (PEFT)** model using **LoRA (Low-Rank Adaptation)**:
        
        1. **Base Model**: T5-small (60M parameters)
        2. **Adapter**: LoRA layers (much smaller, ~1% of parameters)
        3. **Approach**: Only the adapter layers are fine-tuned, then combined with the base model
        
        **Benefits:**
        - Faster training
        - Lower memory usage
        - Easy to share and deploy adapters
        - Maintains base model capabilities
        
        **Files in the model repository:**
        - `adapter_config.json` - LoRA configuration
        - `adapter_model.safetensors` - Adapter weights
        - Tokenizer files
        - No full model weights (that's why direct loading fails)
        """)

if __name__ == "__main__":
    main()
