# Nail Disease Detection and Consultation System

A system that classifies a nail image into one of six categories and then explains the result in plain language using retrieval-augmented generation (RAG) with Mistral-7B. Built as part of a summer research internship at the cHEAL Research Center, PES University (Jun 2025 to Jul 2025). A manuscript based on this work is in preparation.

> This project is for learning and research. It is not a medical device and should not be used to diagnose any condition.

## What it does

1. Takes a photo of a nail as input
2. Classifies it with an EfficientNetV2-S model
3. Retrieves relevant information about the predicted condition from a small knowledge base
4. Uses Mistral-7B to answer the user's question about that condition, grounded in the retrieved text

## Classes

- Healthy Nail
- Onychogryphosis
- Acral Lentiginous Melanoma
- Clubbing
- Pitting
- Blue finger (cyanosis)

## How it works

**Classifier**

- Dataset: [Nail Disease Detection Dataset](https://www.kaggle.com/datasets/nikhilgurav21/nail-disease-detection-dataset) from Kaggle, with the original train and validation folders combined
- Split: 70% train, 15% validation, 15% test (fixed random seed of 42)
- Augmentation: one augmented copy of each training image (rotation, shifts, zoom, horizontal flip)
- Model: EfficientNetV2-S pretrained on ImageNet with the base frozen, followed by global average pooling, dropout, a 128-unit dense layer, dropout and a 6-way softmax
- Training: Adam optimizer (learning rate 0.0001), up to 50 epochs with early stopping and checkpointing on validation loss

**Consultation assistant**

- Prediction Agent: runs the classifier on the uploaded image
- RAG Agent: splits a short disease knowledge file into chunks, embeds them, stores them in FAISS and retrieves the most relevant chunks for the predicted class
- Chat generator: Mistral-7B-Instruct answers the user's question using the retrieved context and the conversation history

## Results

Evaluated on the validation split:

| Metric | Value |
| --- | --- |
| Training accuracy | 92.43% |
| Validation accuracy | 84.32% |

- Healthy nails were separated perfectly in validation (precision and recall of 1.00)
- The weakest classes were blue finger and clubbing, which were most often confused with other conditions
- A held-out test split was created but not evaluated in this notebook

## Tech stack

- Python, TensorFlow and Keras
- EfficientNetV2-S (transfer learning)
- Mistral-7B-Instruct with Hugging Face Transformers
- LangChain and FAISS for retrieval
- scikit-learn, seaborn and matplotlib for evaluation

## Running it

The notebook was built and run in Google Colab with a GPU.

1. Open `nail_disease_detection.ipynb` in Google Colab and select a GPU runtime
2. Add your Kaggle API key (`kaggle.json`) when the notebook asks, so the dataset can be downloaded
3. Log in to Hugging Face when prompted. You need an account with access to Mistral-7B-Instruct. Do not paste tokens into the notebook
4. Run the cells in order

## Limitations

- The knowledge base is a short, hand-written file with one entry per condition, not a clinical source
- The language model can still produce incorrect or overly confident text, so its answers need review
- Accuracy numbers come from the validation split of a single public dataset and may not carry over to real-world photos
- The assistant cells and the generator-based evaluation cells near the end of the notebook rescale pixels by 1/255, which does not match how the classifier was trained. The roughly 26% accuracy printed there reflects that preprocessing mismatch, not the model itself. The 84.32% figure comes from the validation evaluation earlier in the notebook

## Team and context

This was a team project completed during the internship. My contributions were in model development and evaluation, including comparing CNN architectures during model selection.
