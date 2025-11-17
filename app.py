import streamlit as st
import torch
from transformers import pipeline
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
    .model-status {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📝 T5-small LoRA Text Summarization</h1>', unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading summarization model...")
def load_summarizer():
    """Load the T5-small LoRA summarization model from Hugging Face"""
    try:
        # Load the model directly from Hugging Face
        summarizer = pipeline(
            "summarization",
            model="manesh1/t5-small-lora-summarization",
            device=0 if torch.cuda.is_available() else -1  # Use GPU if available
        )
        return summarizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def chunk_text(text, max_chunk_size=512):
    """Split long text into chunks for processing"""
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += sentence + '. '
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + '. '
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def summarize_long_text(summarizer, text, max_length=150, min_length=30):
    """Summarize long text by chunking"""
    if len(text.split()) <= 400:
        # Direct summarization for shorter texts
        result = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        return result[0]['summary_text']
    else:
        # Chunk and summarize for longer texts
        chunks = chunk_text(text)
        summaries = []
        
        progress_bar = st.progress(0)
        for i, chunk in enumerate(chunks):
            result = summarizer(
                chunk,
                max_length=max_length // len(chunks),
                min_length=min_length // len(chunks),
                do_sample=False
            )
            summaries.append(result[0]['summary_text'])
            progress_bar.progress((i + 1) / len(chunks))
        
        # Combine summaries
        combined_summary = " ".join(summaries)
        
        # If combined summary is still long, summarize it again
        if len(combined_summary.split()) > 100:
            final_result = summarizer(
                combined_summary,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            return final_result[0]['summary_text']
        
        return combined_summary

def main():
    # Load model
    with st.spinner("🔄 Loading the T5-small LoRA model from Hugging Face..."):
        summarizer = load_summarizer()
    
    if summarizer is None:
        st.error("❌ Failed to load the model. Please check your internet connection and try again.")
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
        value=150,
        help="Maximum number of words in the summary"
    )
    
    min_length = st.sidebar.slider(
        "Minimum summary length", 
        min_value=10,
        max_value=100,
        value=30,
        help="Minimum number of words in the summary"
    )
    
    # Model info
    st.sidebar.markdown("---")
    st.sidebar.title("ℹ️ Model Information")
    st.sidebar.info("""
    **Model:** T5-small with LoRA adapters  
    **Task:** Text summarization  
    **Source:** Hugging Face Hub  
    **Repository:** [manesh1/t5-small-lora-summarization](https://huggingface.co/manesh1/t5-small-lora-summarization)
    """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 Input Text")
        input_method = st.radio(
            "Choose input method:",
            ["Type text", "Paste text", "Sample text"],
            horizontal=True
        )
        
        input_text = ""
        
        if input_method == "Type text":
            input_text = st.text_area(
                "Enter your text:",
                height=300,
                placeholder="Type the text you want to summarize here...",
                key="input_type"
            )
        elif input_method == "Paste text":
            input_text = st.text_area(
                "Paste your text:",
                height=300,
                placeholder="Paste your text here...",
                key="input_paste"
            )
        else:
            # Sample text
            sample_text = """
            Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. Leading AI textbooks define the field as the study of intelligent agents: any system that perceives its environment and takes actions that maximize its chance of achieving its goals. Some popular accounts use the term artificial intelligence to describe machines that mimic cognitive functions that humans associate with the human mind, such as learning and problem solving.

            AI applications include advanced web search engines, recommendation systems, understanding human speech, self-driving cars, automated decision-making, and competing at the highest level in strategic game systems. As machines become increasingly capable, tasks considered to require intelligence are often removed from the definition of AI, a phenomenon known as the AI effect. For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology.

            Artificial intelligence was founded as an academic discipline in 1956, and in the years since has experienced several waves of optimism, followed by disappointment and the loss of funding, followed by new approaches, success, and renewed funding. AI research has tried and discarded many different approaches during its lifetime, including simulating the brain, modeling human problem solving, formal logic, large databases of knowledge, and imitating animal behavior. In the first decades of the 21st century, highly mathematical and statistical machine learning has dominated the field, and this technique has proved highly successful, helping to solve many challenging problems throughout industry and academia.
            """
            input_text = st.text_area(
                "Sample text (you can modify this):",
                value=sample_text,
                height=300,
                key="input_sample"
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
                        
                        if len(input_text.split()) > 400:
                            summary = summarize_long_text(
                                summarizer, 
                                input_text, 
                                max_length=max_length, 
                                min_length=min_length
                            )
                        else:
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
                            st.metric("Original Length", f"{len(input_text.split())} words")
                        with col2_stat:
                            st.metric("Summary Length", f"{len(summary.split())} words")
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
    
    # Additional information
    st.markdown("---")
    st.subheader("ℹ️ About this Model")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("""
        **Model Features:**
        - Based on T5-small architecture
        - Fine-tuned with LoRA (Low-Rank Adaptation)
        - Optimized for text summarization
        - Efficient parameter usage
        - Handles long documents
        """)
    
    with col_info2:
        st.markdown("""
        **Usage Tips:**
        - For best results, use well-structured text
        - Adjust min/max length for different summary sizes
        - Handles documents up to several thousand words
        - Works with various text domains
        - Automatic chunking for long texts
        """)

if __name__ == "__main__":
    main()
