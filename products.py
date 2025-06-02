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
            similarity = len(common_words) / max(len(product_words), 1)
            
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
    print("DEBUG: Attempting to load dr_ambroziak_embeddings.pkl...")
    try:
        # Try to load from embeddings pickle file
        if os.path.exists('dr_ambroziak_embeddings.pkl'):
            st.info("Trwa próba załadowania głównej bazy produktów (dr_ambroziak_embeddings.pkl). Ten plik jest duży i jego ładowanie może zająć dużo czasu lub zakończyć się niepowodzeniem przy niewystarczających zasobach.")
            with open('dr_ambroziak_embeddings.pkl', 'rb') as f:
                data = pickle.load(f)
                print("DEBUG: Successfully loaded dr_ambroziak_embeddings.pkl.")
                print(f"DEBUG: Type of loaded data: {type(data)}") # This is fine
                if isinstance(data, dict):
                    print(f"DEBUG: Keys in loaded data: {data.keys()}") # This is fine
                elif isinstance(data, list):
                    print(f"DEBUG: Length of loaded list: {len(data)}") # This is fine
            
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
                print(f"DEBUG: First product entry: {products[0]}")
                if isinstance(products[0], dict):
                    print(f"DEBUG: Keys in first product: {products[0].keys()}")
                    # Try to identify a likely embedding key
                    for key, value in products[0].items():
                        if isinstance(value, list) and len(value) > 100: # Heuristic for an embedding
                            print(f"DEBUG: Potential embedding found for key '{key}' with length {len(value)}")
                        elif 'embedding' in key.lower() or 'vector' in key.lower():
                             print(f"DEBUG: Potential embedding found for key '{key}'")
                print(f"INFO: ✅ Wczytano {len(products)} produktów z bazy embeddings")
                return products, True
            else:
                print("WARNING: ⚠️ Plik embeddings nie zawiera danych produktów w oczekiwanym formacie.")
                print("DEBUG: Falling back to get_demo_products().")
                return get_demo_products(), True
                
        else:
            st.warning("Nie znaleziono pliku dr_ambroziak_embeddings.pkl. Używam tymczasowej, demonstracyjnej bazy produktów.")
            print("DEBUG: Falling back to get_demo_products().")
            return get_demo_products(), True
            
    except Exception as e:
        print(f"DEBUG: Exception during pickle loading: {e}") # This is a useful debug print
        st.error(f"Błąd podczas ładowania dr_ambroziak_embeddings.pkl: {e}. Używam tymczasowej, demonstracyjnej bazy produktów.")
        print("DEBUG: Falling back to get_demo_products().")
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
