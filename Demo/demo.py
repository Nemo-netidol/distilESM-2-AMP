import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

model_path = "NakornB/distilESM-2-AMP"

@st.cache_resource
def load_model():
    # Load config first
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        torch_dtype=torch.float32, 
        device_map="cpu",
        ignore_mismatched_sizes=False,
        )
    
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return tokenizer, model

tokenizer, model = load_model()

def is_validate_protein(sequence):
    return bool(re.fullmatch(r'[ACDEFGHIKLMNPQRSTVWY]+', sequence.upper()))

def predict_single(sequence):
    inputs = tokenizer(
        [sequence.upper()],
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    amp_prob = probs[0, 1].item()
    non_amp_prob = probs[0, 0].item()
    preds_class = probs.argmax(dim=1).item()

    label = "AMP" if preds_class == 1 else "Non-AMP"
    confidence = amp_prob if preds_class == 1 else non_amp_prob
    return confidence, label

def predict_batch(df):
    sequences = df['seq']
    amp_probs = []
    labels = []
    for seq in sequences:
        amp_prob, label = predict_single(seq)
        amp_probs.append(amp_prob)
        labels.append(label)

    result_df = pd.concat([df, pd.DataFrame({"class": labels, "confidence": amp_probs})], axis=1)

    return result_df

st.title("🧬 DistilESM-2-AMP: Antimicrobial Peptide Classifier demo")

st.markdown("""
This application demonstrates a machine learning approach to identifying Antimicrobial Peptides (AMPs) from protein sequences. 
**Note:** This is for research demonstration purposes only.
""")

with st.sidebar:
    st.header("👾 Author Information")
    st.write("**Name:** Nakorn Boonprasong")
    st.write("**Contact:** boonprasongnakorn@gmail.com")
    st.divider()
    st.header("📊 Model Metadata")
    st.info(
        """
        - **Architecture**: ESM-2 (Distillation to 3 layers)
        - **Dataset**: UniRef50
        - **Accuracy**: 96.90%
        """
    )

tab1, tab2 = st.tabs(["Single Sequence Prediction", "Batch Prediction (CSV)"])

with tab1:
    st.subheader("Single Sequence Analysis")
    seq_input = st.text_area("Enter Amino Acid Sequence", placeholder="e.g., PROTEIN")    

    if st.button("Predict"):
        if not seq_input.strip():
            st.error("Please enter a sequence.")
        elif not is_validate_protein(seq_input.strip()):
            st.error("Invalid sequence. Please use standard Amino Acid letters (A-Y) without spaces or special characters.")
        else:
            with st.spinner("Analyzing sequence..."):
                col1, col2 = st.columns(2)
                confidence, label = predict_single(seq_input.strip().upper())
                with col1:
                    st.metric("Prediction", label)
                with col2:
                    st.metric("Probability", f'{confidence * 100: .2f}%')
                
                if label == "AMP":
                    st.success("The sequence is classified as AMP")
                else:
                    st.warning("The seqeunce is classified as Non-AMP")

with tab2:
    st.subheader("Batch Analysis via CSV")
    st.markdown("Upload a CSV file with column `seq` that contains peptide sequences.")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

    if uploaded_file is not None:
        st.write("File Preview:")
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head(10))
        required_col = {'seq'}
        if not required_col.issubset(df.columns):
            st.error(f"Your CSV file must contain these column: {required_col}")
        if st.button("Run Prediction"):
            with st.spinner("Processing Prediction..."):
                result_df = predict_batch(df)
                st.divider()
                st.success("Prediction Complete!")
                st.dataframe(result_df)
                # st.sidebar.write(f"AMP: {len(result_df[result_df['class'] == 'AMP'])}")
                # st.sidebar.write(f"Non-AMP: {len(result_df[result_df['class'] == 'Non-AMP'])}")
                
                csv_output = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results as CSV",
                    data=csv_output,
                    file_name="amp_prediction.csv"
                )
            




