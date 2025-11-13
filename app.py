
import streamlit as st
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    BartForConditionalGeneration
)
import requests
import os
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📝",
    layout="wide"
)

@st.cache_resource
def load_model_and_tokenizer():
    """
    Load the model and tokenizer with error handling
    """
    try:
        # First, let's try to determine what type of model this is
        # You might need to adjust this based on your actual model architecture
        repo_path = "manesh230/Encoder-Decoder-T5-BART-Text-Summarization"
        
        # Try to load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(repo_path)
        except:
            # If specific tokenizer fails, try common ones
            try:
                tokenizer = AutoTokenizer.from_pretrained("t5-small")
            except:
                tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
        
        # Try to determine model type and load accordingly
        try:
            # Try loading as T5 first
            model = T5ForConditionalGeneration.from_pretrained(
                repo_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
        except:
            try:
                # Try loading as BART
                model = BartForConditionalGeneration.from_pretrained(
                    repo_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
            except:
                # Fallback to AutoModel
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    repo_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
        
        return model, tokenizer
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def summarize_text(model, tokenizer, text, max_length=150, min_length=30):
    """
    Generate summary for the input text
    """
    try:
        # Prepare input
        if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Tokenize input
        inputs = tokenizer(
            text,
            max_length=1024,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate summary
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        # Decode summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    except Exception as e:
        return f"Error during summarization: {str(e)}"

def main():
    # Header
    st.title("📝 Text Summarization App")
    st.markdown("""
    This app uses a fine-tuned T5/BART model to generate summaries from your text.
    Enter your text below and adjust the parameters as needed.
    """)
    
    # Load model
    with st.spinner("Loading model... This might take a while the first time."):
        model, tokenizer = load_model_and_tokenizer()
    
    if model is None or tokenizer is None:
        st.error("Failed to load the model. Please check the model files and try again.")
        return
    
    # Sidebar for parameters
    st.sidebar.title("Settings")
    max_length = st.sidebar.slider(
        "Maximum summary length",
        min_value=50,
        max_value=300,
        value=150,
        help="Maximum number of tokens in the generated summary"
    )
    
    min_length = st.sidebar.slider(
        "Minimum summary length", 
        min_value=10,
        max_value=100,
        value=30,
        help="Minimum number of tokens in the generated summary"
    )
    
    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["Text Input", "File Upload"]
    )
    
    input_text = ""
    
    if input_method == "Text Input":
        # Text area for input
        input_text = st.text_area(
            "Enter text to summarize:",
            height=200,
            placeholder="Paste your text here...",
            help="Enter the text you want to summarize"
        )
    
    else:
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a text file",
            type=['txt', 'pdf', 'docx'],
            help="Supported formats: TXT, PDF, DOCX"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.type == "text/plain":
                    input_text = str(uploaded_file.read(), "utf-8")
                else:
                    st.warning("Please upload a .txt file for now. PDF and DOCX support coming soon.")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    # Example texts
    with st.expander("💡 Need example text?"):
        examples = [
            "The quick brown fox jumps over the lazy dog. This is a simple example text that demonstrates how the summarization works.",
            """Artificial intelligence is transforming various industries by automating tasks and providing insights from large datasets. 
            Machine learning algorithms can identify patterns and make predictions with high accuracy. Companies are investing heavily 
            in AI research and development to gain competitive advantages. The future of AI looks promising with advancements in 
            natural language processing and computer vision."""
        ]
        
        example_choice = st.selectbox("Choose an example:", ["Example 1", "Example 2"])
        if st.button("Load Example"):
            if example_choice == "Example 1":
                input_text = examples[0]
            else:
                input_text = examples[1]
    
    # Summarize button
    if st.button("Generate Summary", type="primary"):
        if not input_text.strip():
            st.warning("Please enter some text to summarize.")
            return
        
        if len(input_text.strip().split()) < 10:
            st.warning("Text seems too short for meaningful summarization. Please provide longer text.")
            return
        
        with st.spinner("Generating summary..."):
            # Show original text stats
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Original words", len(input_text.split()))
            with col2:
                st.metric("Original characters", len(input_text))
            
            # Generate summary
            summary = summarize_text(model, tokenizer, input_text, max_length, min_length)
            
            # Display summary
            st.subheader("📋 Generated Summary")
            st.write(summary)
            
            # Summary stats
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Summary words", len(summary.split()))
            with col2:
                st.metric("Summary characters", len(summary))
            
            # Compression ratio
            original_words = len(input_text.split())
            summary_words = len(summary.split())
            if original_words > 0:
                compression_ratio = (1 - summary_words / original_words) * 100
                st.metric("Compression", f"{compression_ratio:.1f}%")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Model: [Encoder-Decoder-T5-BART-Text-Summarization](https://github.com/manesh230/Encoder-Decoder-T5-BART-Text-Summarization) | "
        "Built with Streamlit & Transformers"
    )

if __name__ == "__main__":
    main()
