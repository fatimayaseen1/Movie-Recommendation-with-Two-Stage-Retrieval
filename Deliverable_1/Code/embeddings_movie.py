import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
import gc  # Garbage collector
from tqdm import tqdm  # For progress bars

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the dataset
df = pd.read_csv(r"/home/f223090/Ifra zaib/cleaned_movies.csv")
print(f"Dataset loaded with {len(df)} rows")

# Load BERT tokenizer and model
print("Loading BERT model...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model = bert_model.to(device)
bert_model.eval()  # Set to evaluation mode

# Fix column name if needed
if 'Plot Kyeword' in df.columns and 'Plot Keyword' not in df.columns:
    df = df.rename(columns={'Plot Kyeword': 'Plot Keyword'})

# Function to get BERT embeddings in batches to improve efficiency
def get_bert_embeddings_batch(texts, tokenizer, model, batch_size=8):
    """Process texts in batches to get BERT embeddings"""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Handle NaN values
        batch_texts = [str(text) if not pd.isna(text) else "" for text in batch_texts]
        
        # Tokenize
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, 
                          padding=True, max_length=512)
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get CLS token embeddings
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.extend(batch_embeddings)
        
        # Free up memory
        del outputs, inputs
        torch.cuda.empty_cache() if device.type == 'cuda' else None
        
    return all_embeddings

# Process batches with progress bar
print("Processing Overview embeddings...")
overview_texts = df['Overview'].tolist()
overview_embeddings = get_bert_embeddings_batch(
    overview_texts, tokenizer, bert_model, batch_size=16
)
df['overview_embeddings'] = overview_embeddings
print("Overview embeddings completed")

# Free up memory
gc.collect()
torch.cuda.empty_cache() if device.type == 'cuda' else None

print("Processing Plot Keyword embeddings...")
keyword_texts = df['Plot Keyword'].tolist()
keyword_embeddings = get_bert_embeddings_batch(
    keyword_texts, tokenizer, bert_model, batch_size=16
)
df['plot_keywords_embeddings'] = keyword_embeddings
print("Plot Keyword embeddings completed")

# Process genre embeddings
print("Processing Genre embeddings...")
label_encoder = LabelEncoder()
df['genre_encoded'] = label_encoder.fit_transform(df['Generes'])

embedding_dim = 50
genre_embeddings = torch.nn.Embedding(len(label_encoder.classes_), embedding_dim)
genre_embeddings = genre_embeddings.to(device)

# Get genre embeddings
df['genre_embeddings'] = df['genre_encoded'].apply(
    lambda x: genre_embeddings(torch.tensor([x]).to(device)).detach().cpu().numpy().flatten()
)
print("Genre embeddings completed")

# Combine embeddings
print("Combining embeddings...")
def combine_embeddings(row):
    try:
        combined = np.concatenate([
            row['overview_embeddings'],
            row['plot_keywords_embeddings'],
            row['genre_embeddings']
        ])
        return combined
    except Exception as e:
        print(f"Error combining embeddings: {e}")
        # Return zeros array of appropriate size as fallback
        return np.zeros(768 + 768 + embedding_dim)

# Apply the combine function
df['combined_embeddings'] = df.apply(combine_embeddings, axis=1)

# Save embeddings to avoid recomputing
print("Saving embeddings...")
np.save('/home/f223090/Ifra zaib/movie_combined_embeddings.npy', 
        np.array(df['combined_embeddings'].tolist()))

# Print sample embeddings
sample_embeddings = df['combined_embeddings'].head(3).tolist()
for i, vec in enumerate(sample_embeddings):
    print(f"Sample {i+1} embedding vector: {vec[:10]}...")

print("Embedding process completed successfully!")