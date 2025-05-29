import streamlit as st
import pickle
import os
import anthropic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Simple product matching - backward compatibility
def find_matching_products(topic, section_title, products_db, api_key, threshold=0.3):
    """Find products matching content - generator compatibility"""
    if not products_db:
        return []
    
    # Combine topic and section for better matching
    content = f"{topic} {section_title}".lower()
    matching_products = []
    
    for product in products_db:
        # Simple keyword matching
        product_text = f"{product['nazwa']} {product['zastosowanie']}".lower()
        
        # Count matching words
        content_words = set(content.split())
        product_words = set(product_text.split())
        common_words = content_words & product_words
        
        if len(common_words) >= 2:  # At least 2 common words
            similarity = len(common_words) / max(len(product_words), 5)
            
            if similarity >= threshold:
                product_copy = product.copy()
                product_copy['similarity'] = similarity
                # Add opis field for generator compatibility
                product_copy['opis'] = product.get('zastosowanie', 'Brak opisu')
                matching_products.append(product_copy)
    
    # Sort by similarity
    matching_products.sort(key=lambda x: x['similarity'], reverse=True)
    return matching_products[:5]

# Simple analysis function
def analyze_text_for_products(text, produkty_db, api_key=None):
    """Simple analysis that finds product opportunities"""
    if not text or not produkty_db:
        return []
    
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    recommendations = []
    
    for i, paragraph in enumerate(paragraphs):
        # Skip very short paragraphs
        if len(paragraph.split()) < 10:
            continue
        
        # Skip paragraphs that are only about symptoms
        paragraph_lower = paragraph.lower()
        if any(symptom in paragraph_lower for symptom in ['objawy', 'symptomy', 'charakteryzują się', 'pojawiają się']) and \
           not any(solution in paragraph_lower for solution in ['leczenie', 'terapia', 'stosować', 'pomocne', 'warto']):
            continue
        
        # Find matching products for this paragraph
        matches = find_matching_products(paragraph, produkty_db, threshold=0.2)
        
        for product in matches[:2]:  # Max 2 per paragraph
            recommendations.append({
                'paragraph_index': i + 1,
                'paragraph_text': paragraph,
                'product': product,
                'suggestion_type': 'general',
                'main_topics': []
            })
    
    return recommendations

# Product content generation
def generate_product_content(product, content_type, client):
    """Generate content for a product"""
    try:
        if content_type == "opis":
            prompt = f"""
Napisz profesjonalny opis produktu kosmetycznego.

PRODUKT:
- Nazwa: {product['nazwa']}
- Zastosowanie: {product['zastosowanie']}
- Cena: {product.get('cena', 'N/A')}

Napisz zwięzły, atrakcyjny opis (2-3 zdania) podkreślający korzyści i zastosowanie.
Użyj profesjonalnego, ale przystępnego języka.
"""
        elif content_type == "artykul":
            prompt = f"""
Napisz krótki artykuł edukacyjny o produkcie kosmetycznym.

PRODUKT:
- Nazwa: {product['nazwa']}
- Zastosowanie: {product['zastosowanie']}

Napisz artykuł (4-5 akapitów) obejmujący:
- Czym jest produkt i do czego służy
- Jak działa i jakie ma składniki aktywne
- Dla kogo jest przeznaczony
- Jak stosować
- Podsumowanie korzyści

Użyj eksperckiego, ale przystępnego języka.
"""
        else:  # social media post
            prompt = f"""
Napisz angażujący post na social media o produkcie kosmetycznym.

PRODUKT:
- Nazwa: {product['nazwa']}
- Zastosowanie: {product['zastosowanie']}

Napisz krótki, chwytliwy post (2-3 zdania) z:
- Ciekawym hookiem
- Korzyściami produktu
- Call-to-action
- Odpowiednimi emoji

Styl: przyjazny, zachęcający, autentyczny.
"""

        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=500,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content[0].text.strip()
        
    except Exception as e:
        return f"Błąd generowania treści: {e}"

# Product suggestion generation
def generate_product_suggestion(paragraph_text, product, suggestion_type, anthropic_client):
    """Generate contextual product suggestion"""
    
    prompt = f"""
Przeredaguj podany akapit, naturalnie wplatając rekomendację produktu.

ORYGINALNY AKAPIT:
{paragraph_text}

PRODUKT:
- Nazwa: {product['nazwa']}
- Zastosowanie: {product['zastosowanie']}

ZADANIE:
1. NAJPIERW sprawdź czy produkt tematycznie pasuje do treści akapitu
2. Jeśli NIE PASUJE - odpowiedz tylko: "PRODUKT_NIE_PASUJE_DO_KONTEKSTU"
3. Jeśli PASUJE - przeredaguj akapit naturalnie wplatając produkt

WYMAGANIA:
- Zachowaj wszystkie ważne informacje merytoryczne
- Wpleć produkt naturalnie, nie na siłę
- Użyj płynnych przejść językowych
- Zachowaj profesjonalny, edukacyjny ton

Odpowiedz TYLKO przeredagowanym akapitem lub "PRODUKT_NIE_PASUJE_DO_KONTEKSTU".
"""

    try:
        message = anthropic_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=600,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content[0].text.strip()
        
    except Exception as e:
        return f"Błąd generowania sugestii: {e}"

# Filter recommendations
def filter_recommendations_by_quality(recommendations, min_threshold=0.4):
    """Filter out recommendations with very poor thematic relevance"""
    if not recommendations:
        return []
    
    filtered = []
    
    for rec in recommendations:
        relevance = rec['product'].get('thematic_relevance', rec['product'].get('similarity', 0))
        
        if relevance >= min_threshold:
            filtered.append(rec)
    
    return filtered

# Load products database
def load_products_database():
    """Load products database from embeddings file"""
    try:
        # Try to load from embeddings pickle file
        if os.path.exists('dr_ambroziak_embeddings.pkl'):
            with open('dr_ambroziak_embeddings.pkl', 'rb') as f:
                data = pickle.load(f)
            
            # Extract products from embeddings data
            products = []
            
            # Handle different possible structures
            if isinstance(data, dict):
                if 'products' in data:
                    products = data['products']
                elif 'produkty' in data:
                    products = data['produkty']
                else:
                    # Try to extract from embeddings structure
                    for key, value in data.items():
                        if isinstance(value, dict) and 'nazwa' in value:
                            products.append(value)
                        elif isinstance(value, list):
                            products.extend([item for item in value if isinstance(item, dict) and 'nazwa' in item])
            elif isinstance(data, list):
                products = [item for item in data if isinstance(item, dict) and 'nazwa' in item]
            
            if products:
                st.success(f"✅ Wczytano {len(products)} produktów z bazy embeddings")
                return products, True
            else:
                st.warning("⚠️ Plik embeddings nie zawiera danych produktów w oczekiwanym formacie.")
                return get_demo_products(), True
                
        else:
            st.warning("⚠️ Plik dr_ambroziak_embeddings.pkl nie został znaleziony.")
            return get_demo_products(), True
            
    except Exception as e:
        st.error(f"❌ Błąd wczytywania bazy produktów: {e}")
        return get_demo_products(), True

def get_demo_products():
    """Return demo products for testing"""
    return [
        {
            'nazwa': 'Serum Na Przebarwienia Z Kwasem Kojowym',
            'zastosowanie': 'cera trądzikowa, nawilżanie, ochrona słoneczna, rozjaśnianie',
            'cena': '1025,00zł',
            'url': 'https://example.com/serum'
        },
        {
            'nazwa': 'Krem Do Cery Naczynkowej z SPF 15',
            'zastosowanie': 'cera wrażliwa, naczynka, ochrona UV, nawilżanie',
            'cena': '340,00zł', 
            'url': 'https://example.com/krem'
        },
        {
            'nazwa': 'Żel Oczyszczający Przeciwtrądzikowy',
            'zastosowanie': 'oczyszczanie, cera tłusta, trądzik, regulacja sebum',
            'cena': '89,00zł',
            'url': 'https://example.com/zel'
        }
    ]
