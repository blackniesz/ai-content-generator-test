import streamlit as st
import pickle
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ========================================
# PRODUCTS DATABASE
# ========================================

@st.cache_resource
def load_products_database():
    """Load products database with embeddings"""
    try:
        with open('dr_ambroziak_embeddings.pkl', 'rb') as f:
            produkty_db = pickle.load(f)
        return produkty_db, True
    except FileNotFoundError:
        st.warning("⚠️ Nie znaleziono bazy produktów. Aplikacja będzie działać bez rekomendacji.")
        return None, False
    except Exception as e:
        st.error(f"❌ Błąd ładowania bazy produktów: {e}")
        return None, False

def create_query_embedding(text, openai_key):
    """Create embedding for search query"""
    try:
        openai.api_key = openai_key
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Błąd tworzenia embedding: {e}")
        return None

def find_matching_products(topic, section_text, produkty_db, openai_key, top_k=2):
    """Find matching Dr Ambroziak products using AI embeddings"""
    if not produkty_db:
        return []
    
    query_text = f"{topic} {section_text}"
    query_embedding = create_query_embedding(query_text, openai_key)
    
    if query_embedding is None:
        return []
    
    try:
        product_embeddings = []
        product_data = []
        
        for product in produkty_db:
            if 'embedding' in product and product['embedding']:
                product_embeddings.append(product['embedding'])
                product_data.append(product)
        
        if not product_embeddings:
            return []
        
        similarities = cosine_similarity([query_embedding], product_embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        matching_products = []
        for idx in top_indices:
            if similarities[idx] > 0.25:
                product = product_data[idx].copy()
                product['similarity'] = similarities[idx]
                matching_products.append(product)
        
        return matching_products
    except Exception as e:
        st.error(f"Błąd wyszukiwania produktów: {e}")
        return []

def analyze_text_for_products(text, produkty_db, openai_key, similarity_threshold=0.3):
    """Analyze text and find places where products could be mentioned"""
    if not produkty_db:
        return []
    
    # Split text into paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    recommendations = []
    
    for i, paragraph in enumerate(paragraphs):
        # Skip very short paragraphs
        if len(paragraph.split()) < 10:
            continue
            
        # Find matching products for this paragraph
        matching_products = find_matching_products("", paragraph, produkty_db, openai_key, top_k=3)
        
        for product in matching_products:
            if product['similarity'] > similarity_threshold:
                recommendations.append({
                    'paragraph_index': i + 1,
                    'paragraph_text': paragraph[:150] + "..." if len(paragraph) > 150 else paragraph,
                    'product': product,
                    'suggestion_type': determine_suggestion_type(paragraph, product)
                })
    
    return recommendations

def determine_suggestion_type(paragraph_text, product):
    """Determine the best way to mention the product in the paragraph"""
    paragraph_lower = paragraph_text.lower()
    product_category = product.get('kategoria', '').lower()
    product_usage = product.get('zastosowanie', '').lower()
    
    # Determine suggestion type based on context
    if any(word in paragraph_lower for word in ['jak', 'sposób', 'metoda', 'rozwiązanie']):
        return "solution"  # Suggest as a solution
    elif any(word in paragraph_lower for word in ['produkty', 'kosmetyki', 'preparaty']):
        return "direct"    # Direct product mention
    elif any(word in paragraph_lower for word in ['pielęgnacja', 'dbanie', 'troska']):
        return "care"      # Care routine suggestion
    else:
        return "general"   # General mention

def generate_product_suggestion(paragraph_text, product, suggestion_type, anthropic_client):
    """Generate a natural sentence suggesting the product"""
    
    prompt = f"""
    Mam następujący akapit artykułu:
    "{paragraph_text}"
    
    Chcę naturalnie wspomnieć o produkcie Dr Ambroziak:
    - Nazwa: {product['nazwa']}
    - Opis: {product['opis'][:200]}...
    - Zastosowanie: {product['zastosowanie']}
    
    Typ sugestii: {suggestion_type}
    
    Napisz 1-2 zdania, które można NATURALNIE wstawić do tego akapitu, wspominając o produkcie.
    Zdania powinny:
    - Płynnie wpasowywać się w kontekst
    - Nie brzmieć jak reklama
    - Być praktyczne i pomocne
    - Zawierać nazwę produktu
    
    Zwróć TYLKO te 1-2 zdania, bez dodatkowych komentarzy.
    """
    
    try:
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        return f"W pielęgnacji tego typu skóry sprawdzi się {product['nazwa']} - {product['opis'][:100]}..."
