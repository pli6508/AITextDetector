import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import pandas as pd
import fitz  # PyMuPDF for PDF
import docx  # python-docx for DOCX
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = st.secrets["general"]["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)

def check_content(model_name, prompt, input_text):
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": input_text}
        ]
    )
    return completion.choices[0].message.content

model_name = "gpt-4o-mini"
prompt = ("Determine if the following content is written by AI or a human, with your level of confidence in %, "
          "please also reference the portion of content which you think is AI-generated and which portion is human-written. "
          "Your answer should be in this format and each section should start from a new line: \n"
          "Conclusion: [Conclusion here]\n"
          "Confidence Level: AI-generated [confidence%], Human-written [confidence%]\n"
          "Analysis: [Detail which portion is AI-generated and which portion is human-written]")


# Load the tokenizers and models
tokenizer_distilbert = DistilBertTokenizer.from_pretrained('aidenliw/essay-detect-distilbert')
model_distilbert = DistilBertForSequenceClassification.from_pretrained('aidenliw/essay-detect-distilbert')

tokenizer_bert = BertTokenizer.from_pretrained('aidenliw/essay-detect-bert')
model_bert = BertForSequenceClassification.from_pretrained('aidenliw/essay-detect-bert')

tokenizer_roberta = RobertaTokenizer.from_pretrained('aidenliw/essay-detect-roberta')
model_roberta = RobertaForSequenceClassification.from_pretrained('aidenliw/essay-detect-roberta')

# Move the models to the appropriate device (GPU or CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_distilbert.to(device)
model_bert.to(device)
model_roberta.to(device)

# Ensure the models are in evaluation mode
model_distilbert.eval()
model_bert.eval()
model_roberta.eval()

# Define the prediction function
def predict(input_text, tokenizer, model):
    # Tokenize the input text
    encoding = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=256,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    # Move the tensors to the appropriate device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Make the prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    # Get the predicted class
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    
    # Interpret the prediction
    if prediction == 1:
        result = "The content is written by AI."
    else:
        result = "The content is written by a human."
    
    return result

# Define a function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

# Define a function to extract text from a DOCX file
def extract_text_from_docx(docx_file):
    text = ""
    doc = docx.Document(docx_file)
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Initialize session state for results if not already present
if "results" not in st.session_state:
    st.session_state.results = []

# Streamlit interface
st.title("Detect AI-Generated Text")
st.markdown("**Beta Version - Demo Only - Developed by Ping Li, Jianfeng Zhang, Jukai Hu, Aiden Wang, Yang Li**")

# Navigation buttons
st.sidebar.title("Navigation")
dashboard_button = st.sidebar.button("Dashboard")
upload_essay_button = st.sidebar.button("Upload Essay")
batch_analysis_button = st.sidebar.button("Batch Analysis")
results_button = st.sidebar.button("Results")
settings_button = st.sidebar.button("Settings")
help_and_support_button = st.sidebar.button("Help and Support")

# Define the default section to display
if "section" not in st.session_state:
    st.session_state.section = "Dashboard"

# Update the section based on button clicks
if dashboard_button:
    st.session_state.section = "Dashboard"
elif upload_essay_button:
    st.session_state.section = "Upload Essay"
elif batch_analysis_button:
    st.session_state.section = "Batch Analysis"
elif results_button:
    st.session_state.section = "Results"
elif settings_button:
    st.session_state.section = "Settings"
elif help_and_support_button:
    st.session_state.section = "Help and Support"

# Display the selected section
if st.session_state.section == "Dashboard":
    st.header("System Instructions")
    st.write("""
    Welcome to the AI Text Detector!
    
    Here's how you can use this tool:
    
    1. **Upload Essay**:
       - Go to the "Upload Essay" section.
       - Upload your essay file (supports .txt, .docx, .pdf).
       - Alternatively, you can paste the text directly into the provided text area.
       - Click on the "Analyze with DistilBERT", "Analyze with BERT", "Analyze with RoBERTa", or "Analyze with GPT-4o-mini" button to see the result.

    2. **Batch Analysis**:
       - Go to the "Batch Analysis" section.
       - Upload a zip file containing multiple essays.
       - Click on the "Analyze Batch" button to analyze all essays in the batch.

    3. **Results**:
       - Go to the "Results" section to view the analysis results.
       - You can see the status and prediction for each uploaded essay.
       - Use the "Export Results" button to download the results.

    4. **Settings**:
       - Go to the "Settings" section to customize your preferences.
       - Choose your default file upload format and preferred result format.
       - Enable or disable notifications.

    5. **Help and Support**:
       - Go to the "Help and Support" section for a comprehensive user guide.
       - Check the FAQs for common questions.
       - Use the contact form to reach out to support if you need assistance.

    This tool helps you to determine if the content of an essay is AI-generated or human-written, ensuring academic integrity and authenticity.
    """)

elif st.session_state.section == "Upload Essay":
    st.header("Upload an Essay for Analysis")
    
    uploaded_file = st.file_uploader("Choose an essay file (.txt, .docx, .pdf)", type=["txt", "docx", "pdf"])
    essay_text = st.text_area("Or paste the essay text here:")
    
    distilbert_result = st.empty()
    bert_result = st.empty()
    roberta_result = st.empty()
    gpt_result = st.empty()

    if st.button("Analyze with DistilBERT"):
        if uploaded_file is not None:
            if uploaded_file.type == "application/pdf":
                essay_text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                essay_text = extract_text_from_docx(uploaded_file)
            else:
                essay_text = uploaded_file.getvalue().decode("utf-8")
            prediction = predict(essay_text, tokenizer_distilbert, model_distilbert)
            input_display = uploaded_file.name
            distilbert_result.write(f"DistilBERT: {prediction}")
        elif essay_text:
            prediction = predict(essay_text, tokenizer_distilbert, model_distilbert)
            input_display = "Input text - " + " ".join(essay_text.split()[:10]) + "..."
            distilbert_result.write(f"DistilBERT: {prediction}")
        else:
            st.warning("Please upload a file or paste the essay text.")
        
        st.session_state.results.append({
            "Model": "DistilBERT",
            "Input": input_display,
            "Prediction": prediction
        })
        
    if st.button("Analyze with BERT"):
        if uploaded_file is not None:
            if uploaded_file.type == "application/pdf":
                essay_text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                essay_text = extract_text_from_docx(uploaded_file)
            else:
                essay_text = uploaded_file.getvalue().decode("utf-8")
            prediction = predict(essay_text, tokenizer_bert, model_bert)
            input_display = uploaded_file.name
            bert_result.write(f"BERT: {prediction}")
        elif essay_text:
            prediction = predict(essay_text, tokenizer_bert, model_bert)
            input_display = "Input text - " + " ".join(essay_text.split()[:10]) + "..."
            bert_result.write(f"BERT: {prediction}")
        else:
            st.warning("Please upload a file or paste the essay text.")
        
        st.session_state.results.append({
            "Model": "BERT",
            "Input": input_display,
            "Prediction": prediction
        })

    if st.button("Analyze with RoBERTa"):
        if uploaded_file is not None:
            if uploaded_file.type == "application/pdf":
                essay_text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                essay_text = extract_text_from_docx(uploaded_file)
            else:
                essay_text = uploaded_file.getvalue().decode("utf-8")
            prediction = predict(essay_text, tokenizer_roberta, model_roberta)
            input_display = uploaded_file.name
            roberta_result.write(f"RoBERTa: {prediction}")
        elif essay_text:
            prediction = predict(essay_text, tokenizer_roberta, model_roberta)
            input_display = "Input text - " + " ".join(essay_text.split()[:10]) + "..."
            roberta_result.write(f"RoBERTa: {prediction}")
        else:
            st.warning("Please upload a file or paste the essay text.")
        
        st.session_state.results.append({
            "Model": "RoBERTa",
            "Input": input_display,
            "Prediction": prediction
        })

    if st.button("Analyze with GPT-4o-mini"):
        if uploaded_file is not None:
            if uploaded_file.type == "application/pdf":
                essay_text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                essay_text = extract_text_from_docx(uploaded_file)
            else:
                essay_text = uploaded_file.getvalue().decode("utf-8")
            prediction = check_content(model_name, prompt, essay_text)
            input_display = uploaded_file.name
            gpt_result.write(f"GPT-4o-mini: {prediction}")
        elif essay_text:
            prediction = check_content(model_name, prompt, essay_text)
            input_display = "Input text - " + " ".join(essay_text.split()[:10]) + "..."
            gpt_result.write(f"GPT-4o-mini: {prediction}")
        else:
            st.warning("Please upload a file or paste the essay text.")
        
        st.session_state.results.append({
            "Model": "GPT-4o-mini",
            "Input": input_display,
            "Prediction": prediction
        })

elif st.session_state.section == "Batch Analysis":
    st.header("Batch Analysis")
    
    batch_file = st.file_uploader("Upload a zip file containing multiple essays", type="zip")
    
    if st.button("Analyze Batch"):
        if batch_file is not None:
            # Code to handle batch file upload and analysis
            st.write("Analyzing batch of essays...")
            
            # Display progress indicator (dummy example)
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
            st.success("Batch analysis completed.")
        else:
            st.warning("Please upload a zip file.")

elif st.session_state.section == "Results":
    st.header("Analysis Results")
    
    if st.session_state.results:
        results_df = pd.DataFrame(st.session_state.results)
        st.table(results_df)
        
        if st.button("Export Results"):
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name='analysis_results.csv',
                mime='text/csv',
            )
    else:
        st.write("No results to display.")

elif st.session_state.section == "Settings":
    st.header("Settings")
    
    st.subheader("User Preferences")
    file_format = st.selectbox("Default file upload format", ["txt", "docx", "pdf"])
    result_format = st.selectbox("Preferred result format", ["Table", "Chart"])
    notifications = st.checkbox("Enable notifications")
    
    st.subheader("Model Configuration")
    model_choice = st.selectbox("Choose model", ["DistilBERT", "BERT", "RoBERTa"])
    
    if st.button("Save Settings"):
        # Code to save settings
        st.success("Settings saved.")
    
    st.subheader("System Logs")
    if st.button("View Logs"):
        # Code to display system logs
        st.write("Displaying system logs...")

elif st.session_state.section == "Help and Support":
    st.header("Help and Support")
    
    st.subheader("User Guide")
    st.write("A comprehensive guide on how to use the system.")
    
    st.subheader("FAQs")
    st.write("Frequently Asked Questions:")
    st.write("Q: How to upload an essay?\nA: Use the 'Upload Essay' section to upload or paste your essay.")
    
    st.subheader("Contact Support")
    st.write("If you have any questions or need assistance, please contact us.")
    contact_form = st.form("contact_form")
    contact_form.text_input("Your Name")
    contact_form.text_input("Your Email")
    contact_form.text_area("Your Message")
    if contact_form.form_submit_button("Submit"):
        # Code to handle form submission
        st.success("Message sent to support.")
