import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import anthropic
import pandas as pd
import streamlit as st
import pickle
import os

# Klasyfikacja typów akapitów
PARAGRAPH_TYPES = {
    'symptoms_only': {
        'keywords': ['objawy', 'symptomy', 'występują', 'pojawiają się', 'manifestują się', 'charakteryzują się',
                    'można zaobserwować', 'widoczne są', 'objawia się', 'wykazuje', 'cechuje się'],
        'exclude_keywords': ['leczenie', 'terapia', 'stosować', 'aplikować', 'używać', 'polecane', 'warto', 'można zastosować'],
        'should_recommend': False
    },
    'treatment': {
        'keywords': ['leczenie', 'terapia', 'zabieg', 'procedura', 'aplikacja', 'stosowanie', 'używanie',
                    'można zastosować', 'warto użyć', 'polecane', 'skuteczne', 'pomocne'],
        'exclude_keywords': ['objawy które', 'symptomy takie jak', 'charakteryzuje się'],
        'should_recommend': True
    },
    'skincare': {
        'keywords': ['pielęgnacja', 'codziennie', 'regularnie', 'rano', 'wieczorem', 'kremy', 'serum',
                    'preparaty', 'kosmetyki', 'nawilżanie', 'oczyszczanie', 'rutyna'],
        'exclude_keywords': [],
        'should_recommend': True
    },
    'prevention': {
        'keywords': ['zapobieganie', 'profilaktyka', 'unikanie', 'ochrona', 'prewencja', 'aby uniknąć',
                    'w celu zapobieżenia', 'chroniąc', 'zabezpieczając'],
        'exclude_keywords': [],
        'should_recommend': True
    },
    'advice': {
        'keywords': ['warto', 'polecane', 'zalecane', 'sugerowane', 'można', 'dobrze jest',
                    'specjaliści zalecają', 'dermatologowie polecają', 'eksperci sugerują'],
        'exclude_keywords': ['objawy', 'symptomy'],
        'should_recommend': True
    }
}

# Tematy kosmetyczne z kontekstem zastosowania
SKINCARE_TOPICS = {
    'acne': {
        'keywords': ['trądzik', 'wypryski', 'zaskórniki', 'pory', 'sebum', 'łojotok', 'pryszcze'],
        'treatment_keywords': ['oczyszczanie', 'złuszczanie', 'kwasy', 'antybakteryjne', 'regulacja'],
        'related_products': ['żel oczyszczający', 'toner', 'serum', 'kwas', 'peeling']
    },
    'aging': {
        'keywords': ['starzenie', 'zmarszczki', 'anti-age', 'kolagen', 'elastyna', 'ujędrnienie'],
        'treatment_keywords': ['liftingujące', 'przeciwzmarszczkowe', 'regenerujące', 'ujędrniające'],
        'related_products': ['serum', 'krem', 'maska', 'retinol', 'witamina c']
    },
    'pigmentation': {
        'keywords': ['przebarwienia', 'plamy', 'melanina', 'hiperpigmentacja', 'rozjaśnienie'],
        'treatment_keywords': ['rozjaśniające', 'wybielające', 'przeciwplamowe', 'wyrównujące koloryt'],
        'related_products': ['serum rozjaśniające', 'krem przeciwplamowy', 'kwas kojowy']
    },
    'sensitivity': {
        'keywords': ['wrażliwa', 'podrażnienia', 'alergiczne', 'swędzenie', 'zaczerwienienie'],
        'treatment_keywords': ['łagodzące', 'uspokajające', 'hipoalergiczne', 'delikatne'],
        'related_products': ['łagodzący', 'uspokajający', 'delikatny', 'hipoalergiczny']
    },
    'dryness': {
        'keywords': ['sucha', 'odwodnienie', 'łuszczenie', 'szorstka', 'napięcie skóry'],
        'treatment_keywords': ['nawilżające', 'odżywiające', 'regenerująca', 'natłuszczające'],
        'related_products': ['nawilżający', 'odżywczy', 'regenerujący', 'masło', 'olejek']
    },
    'rosacea': {
        'keywords': ['rumień', 'naczynka', 'zaczerwienienia', 'kuperwata', 'teleangiektazje'],
        'treatment_keywords': ['wzmacniające naczynia', 'przeciwzaczerwienieowe', 'łagodzące'],
        'related_products': ['wzmacniający naczynia', 'przeciwzapalny', 'łagodzący']
    }
}

def analyze_text_for_products(text, produkty_db, anthropic_api_key):
    """Enhanced analysis that understands paragraph context using Claude"""
    
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    recommendations = []
    
    for i, paragraph in enumerate(paragraphs):
        # Skip very short paragraphs
        if len(paragraph.split()) < 15:
            continue
        
        # Classify paragraph type
        paragraph_type = classify_paragraph_type(paragraph)
        
        # Skip paragraphs that are only about symptoms/problems
        if not paragraph_type['should_recommend']:
            print(f"Skipping paragraph {i+1}: {paragraph_type['type']} (symptoms only)")
            continue
        
        # Find skincare topics in paragraph
        paragraph_topics = identify_treatment_context(paragraph)
        
        # Skip paragraphs without clear treatment/care context
        if not paragraph_topics:
            print(f"Skipping paragraph {i+1}: No treatment context found")
            continue
        
        # Find products that match both topic and context
        relevant_products = find_contextually_relevant_products(
            paragraph, paragraph_topics, paragraph_type, produkty_db
        )
        
        # Add only high-quality matches
        for product in relevant_products[:2]:  # Max 2 products per paragraph
            if product['contextual_relevance'] >= 0.7:  # Higher threshold
                recommendations.append({
                    'paragraph_index': i + 1,
                    'paragraph_text': paragraph,
                    'product': product,
                    'suggestion_type': paragraph_type['type'],
                    'main_topics': paragraph_topics,
                    'context_type': paragraph_type['type']
                })
    
    return recommendations

def classify_paragraph_type(paragraph):
    """Classify paragraph by its intent and context"""
    paragraph_lower = paragraph.lower()
    
    best_match = {
        'type': 'unknown',
        'score': 0,
        'should_recommend': False
    }
    
    for para_type, data in PARAGRAPH_TYPES.items():
        # Count positive keywords
        positive_matches = sum(1 for keyword in data['keywords'] if keyword in paragraph_lower)
        
        # Count exclusion keywords (negative points)
        negative_matches = sum(1 for keyword in data['exclude_keywords'] if keyword in paragraph_lower)
        
        # Calculate score
        score = positive_matches - (negative_matches * 1.5)  # Negative keywords are weighted more
        
        if score > best_match['score']:
            best_match = {
                'type': para_type,
                'score': score,
                'should_recommend': data['should_recommend']
            }
    
    # Special logic: if paragraph mentions symptoms AND solutions, it's treatment
    if ('objawy' in paragraph_lower or 'symptomy' in paragraph_lower) and \
       any(word in paragraph_lower for word in ['można', 'warto', 'stosować', 'pomocne', 'leczenie']):
        best_match['should_recommend'] = True
        best_match['type'] = 'treatment'
    
    return best_match

def identify_treatment_context(paragraph):
    """Identify skincare topics with treatment/care context"""
    paragraph_lower = paragraph.lower()
    identified_topics = []
    
    for topic, data in SKINCARE_TOPICS.items():
        # Check for topic keywords
        topic_matches = sum(1 for keyword in data['keywords'] if keyword in paragraph_lower)
        
        # More importantly - check for treatment context
        treatment_matches = sum(1 for keyword in data['treatment_keywords'] if keyword in paragraph_lower)
        
        # Must have both topic AND treatment context
        if topic_matches >= 1 and treatment_matches >= 1:
            strength = (topic_matches + treatment_matches * 1.5) / (len(data['keywords']) + len(data['treatment_keywords']))
            identified_topics.append({
                'topic': topic,
                'strength': min(strength, 1.0),
                'has_treatment_context': True
            })
        elif topic_matches >= 2:  # Very strong topic match might be OK without explicit treatment words
            # But check it's not just listing symptoms
            if not any(symptom_word in paragraph_lower for symptom_word in ['objawy', 'symptomy', 'charakteryzują się', 'pojawiają się']):
                strength = topic_matches / len(data['keywords'])
                identified_topics.append({
                    'topic': topic,
                    'strength': strength,
                    'has_treatment_context': False
                })
    
    return identified_topics

def find_contextually_relevant_products(paragraph, paragraph_topics, paragraph_type, produkty_db):
    """Find products that match both topic and usage context"""
    
    relevant_products = []
    
    for product in produkty_db:
        max_relevance = 0
        
        for topic_info in paragraph_topics:
            topic = topic_info['topic']
            topic_strength = topic_info['strength']
            
            # Calculate product relevance for this topic
            product_relevance = calculate_contextual_product_relevance(
                product, topic, paragraph, paragraph_type
            )
            
            # Boost score if paragraph has strong treatment context
            if topic_info['has_treatment_context']:
                product_relevance *= 1.2
            
            # Weight by topic strength in paragraph
            weighted_relevance = product_relevance * topic_strength
            max_relevance = max(max_relevance, weighted_relevance)
        
        if max_relevance > 0.5:  # Minimum threshold
            product_copy = product.copy()
            product_copy['contextual_relevance'] = max_relevance
            product_copy['thematic_relevance'] = max_relevance  # For backward compatibility
            relevant_products.append(product_copy)
    
    # Sort by relevance
    relevant_products.sort(key=lambda x: x['contextual_relevance'], reverse=True)
    
    return relevant_products[:3]

def calculate_contextual_product_relevance(product, topic, paragraph, paragraph_type):
    """Calculate how well a product matches topic AND context"""
    
    product_text = f"{product['nazwa']} {product['zastosowanie']}".lower()
    paragraph_lower = paragraph.lower()
    topic_data = SKINCARE_TOPICS[topic]
    
    # 1. Topic keyword matching
    topic_score = 0
    for keyword in topic_data['keywords']:
        if keyword in product_text:
            topic_score += 0.2
    
    # 2. Treatment keyword matching (more important)
    treatment_score = 0
    for keyword in topic_data['treatment_keywords']:
        if keyword in product_text:
            treatment_score += 0.3
    
    # 3. Product type matching
    product_type_score = 0
    for product_type in topic_data['related_products']:
        if product_type in product_text:
            product_type_score += 0.2
    
    # 4. Context appropriateness
    context_score = 0
    if paragraph_type['type'] == 'treatment':
        # For treatment contexts, prefer active ingredients
        if any(word in product_text for word in ['kwas', 'serum', 'aktywny', 'intensywny']):
            context_score += 0.3
    elif paragraph_type['type'] == 'skincare':
        # For skincare contexts, prefer gentle care products
        if any(word in product_text for word in ['delikatny', 'codziennego użytku', 'pielęgnujący']):
            context_score += 0.3
    elif paragraph_type['type'] == 'prevention':
        # For prevention, prefer protective products
        if any(word in product_text for word in ['ochrona', 'spf', 'zabezpieczający']):
            context_score += 0.3
    
    # 5. Semantic similarity with paragraph
    semantic_score = calculate_semantic_similarity(paragraph_lower, product_text) * 0.2
    
    # Calculate total score with different weights
    total_score = (
        topic_score * 0.2 +           # Topic match
        treatment_score * 0.4 +       # Treatment context (most important)
        product_type_score * 0.2 +    # Product type
        context_score * 0.15 +        # Usage context
        semantic_score * 0.05         # General similarity
    )
    
    return min(total_score, 1.0)

def calculate_semantic_similarity(text1, text2):
    """Calculate semantic similarity between two texts"""
    try:
        vectorizer = TfidfVectorizer(stop_words=None, max_features=1000)
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity_matrix = cosine_similarity(tfidf_matrix)
        return similarity_matrix[0][1]
    except:
        return 0.0

# Backward compatibility functions for generator.py and app.py
def load_products_database():
    """Load products database from embeddings file - backward compatibility function"""
    try:
        # Try to load from embeddings pickle file
        if os.path.exists('dr_ambroziak_embbedings.pkl'):
            with open('dr_ambroziak_embbedings.pkl', 'rb') as f:
                data = pickle.load(f)
            
            # Extract products from embeddings data
            products = []
            
            # Handle different possible structures of the pickle file
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
            st.warning("⚠️ Plik dr_ambroziak_embbedings.pkl nie został znaleziony.")
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
def find_matching_products(content, products_db, threshold=0.3):
    """Find products matching content - backward compatibility function"""
    if not content or not products_db:
        return []
    
    matching_products = []
    content_lower = content.lower()
    
    for product in products_db:
        # Simple keyword matching for backward compatibility
        product_text = f"{product['nazwa']} {product['zastosowanie']}".lower()
        
        # Calculate basic similarity
        similarity = calculate_semantic_similarity(content_lower, product_text)
        
        if similarity >= threshold:
            product_copy = product.copy()
            product_copy['similarity'] = similarity
            matching_products.append(product_copy)
    
    # Sort by similarity
    matching_products.sort(key=lambda x: x['similarity'], reverse=True)
    return matching_products[:5]  # Return top 5

def generate_product_content(product, content_type, client):
    """Generate content for a product - backward compatibility function"""
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
            model="claude-3-haiku-20240307",
            max_tokens=500,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content[0].text.strip()
        
    except Exception as e:
        return f"Błąd generowania treści: {e}"

def generate_product_suggestion(paragraph_text, product, suggestion_type, anthropic_client):
    """Generate contextual product suggestion with better context understanding"""
    
    # First, let Claude analyze if the product makes sense in this context
    context_check_prompt = f"""
Przeanalizuj czy podany produkt kosmetyczny pasuje do kontekstu tego akapitu.

AKAPIT:
{paragraph_text}

PRODUKT:
- Nazwa: {product['nazwa']}
- Zastosowanie: {product['zastosowanie']}

PYTANIE: Czy ten produkt ma sens w kontekście tego akapitu?

Odpowiedz tylko "TAK" jeśli:
- Akapit mówi o leczeniu, pielęgnacji lub rozwiązaniach
- Produkt pasuje tematycznie do problemu
- Rekomendacja byłaby naturalna i pomocna

Odpowiedz "NIE" jeśli:
- Akapit tylko opisuje objawy/symptomy
- Produkt nie pasuje do tematu
- Wstawienie byłoby wymuszone

Odpowiedz jednym słowem: TAK lub NIE
"""

    try:
        # Check context appropriateness first
        context_check = anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            temperature=0.1,
            messages=[{"role": "user", "content": context_check_prompt}]
        )
        
        if "NIE" in context_check.content[0].text.upper():
            return "PRODUKT_NIE_PASUJE_DO_KONTEKSTU"
        
        # If context is appropriate, generate the suggestion
        integration_styles = {
            'treatment': 'jako element terapii lub leczenia',
            'skincare': 'jako część codziennej pielęgnacji',
            'prevention': 'jako środek zapobiegawczy',
            'advice': 'jako rekomendowaną opcję',
            'symptoms_only': 'nie integruj - tylko objawy'
        }
        
        style_instruction = integration_styles.get(suggestion_type, 'w sposób naturalny')
        
        main_prompt = f"""
Przeredaguj podany akapit, naturalnie wplatając rekomendację produktu {style_instruction}.

ORYGINALNY AKAPIT:
{paragraph_text}

PRODUKT:
- Nazwa: {product['nazwa']}
- Zastosowanie: {product['zastosowanie']}

WYMAGANIA:
1. Zachowaj wszystkie ważne informacje merytoryczne z oryginału
2. Wpleć produkt w sposób naturalny, nie na siłę
3. Użyj płynnych przejść językowych (np. "W takich przypadkach pomocne może być...", "Warto rozważyć...", "Dobrym rozwiązaniem jest...")
4. Zachowaj profesjonalny, edukacyjny ton
5. Nie zmieniaj drastycznie długości tekstu

PRZYKŁAD DOBREJ INTEGRACJI:
Oryginalny: "Laser frakcyjny pomaga niwelować przebarwienia po trądziku."
Zmodyfikowany: "Laser frakcyjny pomaga niwelować przebarwienia po trądziku. W uzupełnieniu zabiegów profesjonalnych warto rozważyć domową pielęgnację preparatami takimi jak {product['nazwa']}, które {product['zastosowanie']}."

Odpowiedz TYLKO przeredagowanym akapitem.
"""

        message = anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=600,
            temperature=0.2,
            messages=[{"role": "user", "content": main_prompt}]
        )
        
        return message.content[0].text.strip()
        
    except Exception as e:
        return f"Błąd generowania sugestii: {e}"
