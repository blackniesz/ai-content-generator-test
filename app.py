import streamlit as st
import anthropic
import openai
import requests
from bs4 import BeautifulSoup
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import time
import json

# ========================================
# KONFIGURACJA STRONY
# ========================================

st.set_page_config(
    page_title="AI Content Generator - Dr Ambroziak",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# STYLE CSS
# ========================================

st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stat-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .editor-header {
        background: linear-gradient(90deg, #2E86AB, #A23B72);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# INICJALIZACJA SESSION STATE
# ========================================

if 'generated_article' not in st.session_state:
    st.session_state.generated_article = ""
if 'edited_article' not in st.session_state:
    st.session_state.edited_article = ""
if 'article_history' not in st.session_state:
    st.session_state.article_history = []
if 'products_loaded' not in st.session_state:
    st.session_state.products_loaded = False
if 'produkty_db' not in st.session_state:
    st.session_state.produkty_db = None

# ========================================
# FUNKCJE API I EMBEDDINGS
# ========================================

@st.cache_resource
def load_api_keys():
    """Load API keys from Streamlit secrets"""
    try:
        return {
            'anthropic': st.secrets["ANTHROPIC_API_KEY"],
            'openai': st.secrets["OPENAI_API_KEY"], 
            'google_api': st.secrets["GOOGLE_API_KEY"],
            'google_cx': st.secrets["GOOGLE_CX"]
        }
    except KeyError as e:
        st.error(f"Brak klucza API: {e}")
        return None

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

# ========================================
# FUNKCJE WORKFLOW
# ========================================

def search_competition(topic, google_api_key, google_cx):
    """Search and analyze competition articles"""
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'key': google_api_key,
        'cx': google_cx,
        'q': f'"{topic}" artykuł blog',
        'num': 8
    }
    
    try:
        response = requests.get(search_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        competition = []
        if 'items' in data:
            for item in data['items']:
                competition.append({
                    'title': item.get('title', ''),
                    'url': item.get('link', ''),
                    'snippet': item.get('snippet', '')
                })
        
        return competition
    except Exception as e:
        st.error(f"Błąd wyszukiwania konkurencji: {e}")
        return []

def analyze_competition_length(competition):
    """Analyze length of competition articles"""
    lengths = []
    
    for article in competition[:5]:
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(article['url'], headers=headers, timeout=8)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.decompose()
            
            content = ""
            for selector in ['article', 'main', '.content', '.post-content', '.entry-content']:
                elements = soup.select(selector)
                if elements:
                    content = ' '.join([elem.get_text(strip=True) for elem in elements])
                    break
            
            if not content:
                content = soup.get_text(strip=True)
            
            word_count = len(content.split())
            if 500 < word_count < 5000:
                lengths.append(word_count)
        except:
            continue
    
    if lengths:
        average = sum(lengths) / len(lengths)
        target = int(average * 1.2)
        return target, average, lengths
    else:
        return 2000, 0, []

def search_information(topic, google_api_key, google_cx, limit=6):
    """Search for information about the topic"""
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'key': google_api_key,
        'cx': google_cx,
        'q': topic,
        'num': limit
    }
    
    try:
        response = requests.get(search_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        results = []
        if 'items' in data:
            for item in data['items']:
                results.append({
                    'title': item.get('title', ''),
                    'url': item.get('link', ''),
                    'snippet': item.get('snippet', '')
                })
        
        return results
    except Exception as e:
        st.error(f"Błąd wyszukiwania informacji: {e}")
        return []

def extract_page_content(url, title, snippet):
    """Extract content from a webpage"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        content = ""
        for selector in ['article', 'main', '.content', '.post-content', '.entry-content', 'p']:
            elements = soup.select(selector)
            if elements:
                content = ' '.join([elem.get_text(strip=True) for elem in elements])
                break
        
        if not content:
            content = soup.get_text(strip=True)
        
        content = ' '.join(content.split())[:2500]
        return f"{snippet}\n\n{content}" if content else snippet
    except:
        return f"{snippet}\n\nBrak dostępu do pełnej treści strony."

def analyze_facts(content, topic, anthropic_key):
    """Analyze facts using Claude"""
    client = anthropic.Anthropic(api_key=anthropic_key)
    
    prompt = f"""
    Przeanalizuj poniższe treści dotyczące tematu "{topic}" i wyciągnij najważniejsze fakty.
    
    Treści:
    {content}
    
    Przedstaw fakty w formie listy punktów. Skup się na:
    - Kluczowych definicjach i objawach
    - Przyczynach problemu
    - Skutecznych metodach leczenia
    - Praktycznych wskazówkach
    - Statystykach i danych naukowych
    
    Format odpowiedzi: krótkie, konkretne punkty w stylu lifestyle'owym, przyjazne dla czytelnika.
    """
    
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        st.error(f"Błąd analizy faktów: {e}")
        return ""

def create_outline(topic, facts, target_words, anthropic_key):
    """Create article outline"""
    client = anthropic.Anthropic(api_key=anthropic_key)
    
    prompt = f"""
    Na podstawie poniższych faktów o temacie "{topic}", stwórz konspekt artykułu lifestyle'owego zoptymalizowanego pod SEO.

    Fakty:
    {facts}

    WYMAGANIA:
    1. Tytuł główny (H1) - atrakcyjny, SEO-friendly, zawierający główne słowo kluczowe
    2. Krótki wstęp (3-5 zdań) - zajawka bez użycia zwrotów typu "w tym artykule"
    3. 5-7 śródtytułów (H2) - zróżnicowanych, zawierających long-tail keywords
    4. Dla każdego H2 krótki opis treści (1-2 zdania)
    5. Cel: około {target_words} słów w całym artykule
    6. Styl: przyjazny, lifestyle'owy, praktyczny, user-friendly

    Format markdown:
    # Główny tytuł H1
    
    [Wstęp 3-5 zdań bez "w tym artykule"]
    
    ## 1. Pierwszy śródtytuł H2
    Opis: Co będzie w tej sekcji...
    
    ## 2. Drugi śródtytuł H2  
    Opis: Co będzie w tej sekcji...
    
    STRUKTURA POWINNA BYĆ UROZMAICONA - niektóre sekcje mogą zawierać listy punktowane tam gdzie to zasadne.
    """
    
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        st.error(f"Błąd tworzenia konspektu: {e}")
        return ""

def write_section(topic, outline, facts, section_title, written_sections, remaining_sections, target_words, matching_products, anthropic_key):
    """Write article section"""
    client = anthropic.Anthropic(api_key=anthropic_key)
    
    products_info = ""
    if matching_products:
        products_info = "\n\nDOSTĘPNE PRODUKTY DR AMBROZIAK (rekomenduj subtelnie gdzie pasuje):\n"
        for product in matching_products:
            similarity_info = f" (podobieństwo: {product['similarity']:.1%})" if 'similarity' in product else ""
            products_info += f"- {product['nazwa']}: {product['opis'][:150]}... - {product['zastosowanie']} - {product['url']}{similarity_info}\n"
    
    section_target = target_words // len(re.findall(r'## (?:\d+\.\s*)?(.+)', outline)) if outline else 300
    
    prompt = f"""
    Napisz treść sekcji artykułu lifestyle'owego dla sklepu drambroziak.com.

    TEMAT GŁÓWNY: {topic}
    ŚRÓDTYTUŁ SEKCJI: {section_title}
    CEL SŁÓW CAŁEGO ARTYKUŁU: {target_words}
    CEL SŁÓW TEJ SEKCJI: {section_target}

    KONSPEKT CAŁEGO ARTYKUŁU:
    {outline}

    DOSTĘPNE FAKTY:
    {facts}

    CO JUŻ ZOSTAŁO NAPISANE:
    {written_sections if written_sections else "To jest pierwsza sekcja"}

    CO BĘDZIE NAPISANE PÓŹNIEJ:
    {remaining_sections if remaining_sections else "To jest ostatnia sekcja"}
    
    {products_info}

    WYMAGANIA:
    - Napisz odpowiednią ilość treści (patrz cel słów dla sekcji)
    - Styl: przyjazny, lifestyle'owy, praktyczny, user-friendly
    - NIE powtarzaj informacji z już napisanych sekcji
    - NIE wyprzedzaj treści z przyszłych sekcji
    - Używaj konkretnych faktów z dostępnych danych
    - STRUKTURA: akapity + listy punktowane tam gdzie zasadne (dla lepszej czytelności)
    - Jeśli znajdziesz naturalne miejsce, subtelnie wspomniej o produkcie Dr Ambroziak (bez forsowania!)
    - Format: markdown (bez nagłówka H2 - zostanie dodany automatycznie)
    - Używaj pogrubień **tekst** dla kluczowych pojęć
    
    Zwróć TYLKO treść sekcji, bez dodatkowych komentarzy.
    """
    
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        st.error(f"Błąd pisania sekcji: {e}")
        return ""

# ========================================
# GŁÓWNY WORKFLOW
# ========================================

def generate_article(topic, target_words, api_keys, produkty_db, products_loaded):
    """Main article generation workflow"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Competition analysis
    status_text.text("🔍 Analizuję konkurencję...")
    progress_bar.progress(10)
    
    competition = search_competition(topic, api_keys['google_api'], api_keys['google_cx'])
    
    # Step 2: Information gathering
    status_text.text("📚 Zbiera informacje...")
    progress_bar.progress(25)
    
    search_results = search_information(topic, api_keys['google_api'], api_keys['google_cx'])
    all_content = ""
    
    for i, result in enumerate(search_results):
        content = extract_page_content(result['url'], result['title'], result['snippet'])
        all_content += f"\n--- Źródło {i+1}: {result['title']} ---\n{content}\n"
    
    # Step 3: Fact analysis
    status_text.text("🤖 Analizuję fakty przez Claude...")
    progress_bar.progress(40)
    
    facts = analyze_facts(all_content, topic, api_keys['anthropic'])
    
    # Step 4: Create outline
    status_text.text("📋 Tworzę konspekt...")
    progress_bar.progress(55)
    
    outline = create_outline(topic, facts, target_words, api_keys['anthropic'])
    
    # Parse outline
    title_match = re.search(r'# (.+)', outline)
    title = title_match.group(1) if title_match else topic
    
    intro_match = re.search(r'# .+?\n\n(.+?)\n\n##', outline, re.DOTALL)
    intro = intro_match.group(1).strip() if intro_match else ""
    
    section_titles = re.findall(r'## (?:\d+\.\s*)?(.+)', outline)
    
    # Step 5: Write sections
    sections = []
    written_content = ""
    
    for i, section_title in enumerate(section_titles):
        status_text.text(f"✍️ Piszę sekcję {i+1}/{len(section_titles)}: {section_title}")
        progress_bar.progress(60 + (30 * i / len(section_titles)))
        
        remaining_sections = "\n".join([f"## {s}" for s in section_titles[i+1:]])
        
        # Find matching products for this section
        matching_products = []
        if products_loaded and produkty_db:
            matching_products = find_matching_products(topic, section_title, produkty_db, api_keys['openai'])
        
        section_content = write_section(
            topic, outline, facts, section_title, written_content, 
            remaining_sections, target_words, matching_products, api_keys['anthropic']
        )
        
        section_with_title = f"## {section_title}\n\n{section_content}"
        sections.append(section_with_title)
        written_content += section_with_title + "\n\n"
    
    # Step 6: Finalize article
    status_text.text("📄 Finalizuję artykuł...")
    progress_bar.progress(95)
    
    final_article = f"# {title}\n\n"
    if intro:
        final_article += f"{intro}\n\n"
    final_article += "\n\n".join(sections)
    
    # Add Dr Ambroziak promotion if needed
    if "dr ambroziak" not in final_article.lower() and products_loaded:
        final_article += f"\n\n---\n\n**Profesjonalna pielęgnacja skóry** to podstawa zdrowia i piękna. Jeśli szukasz skutecznych kosmetyków opartych na najnowszych osiągnięciach dermatologii, sprawdź [ofertę Dr Ambroziak Laboratorium](https://drambroziak.com) - produkty stworzone przez ekspertów z ponad 20-letnim doświadczeniem."
    
    progress_bar.progress(100)
    status_text.text("✅ Artykuł gotowy!")
    
    return final_article

# ========================================
# MAIN STREAMLIT APP
# ========================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 AI Content Generator - Dr Ambroziak</h1>', unsafe_allow_html=True)
    
    # Load API keys
    api_keys = load_api_keys()
    if not api_keys:
        st.error("Brak konfiguracji kluczy API. Skontaktuj się z administratorem.")
        return
    
    # Load products database
    if not st.session_state.products_loaded:
        produkty_db, products_loaded = load_products_database()
        st.session_state.produkty_db = produkty_db
        st.session_state.products_loaded = products_loaded
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Ustawienia")
        
        target_words = st.slider(
            "🎯 Długość artykułu (słowa)",
            min_value=1000,
            max_value=5000,
            value=2000,
            step=250,
            help="Docelowa liczba słów w artykule"
        )
        
        st.markdown("---")
        
        # Products status
        if st.session_state.products_loaded:
            st.success(f"✅ Baza produktów: {len(st.session_state.produkty_db)} produktów")
        else:
            st.warning("⚠️ Baza produktów: niedostępna")
        
        st.markdown("---")
        
        # Article history
        if st.session_state.article_history:
            st.subheader("📚 Historia artykułów")
            for i, (topic, timestamp) in enumerate(st.session_state.article_history[-5:]):
                st.text(f"{i+1}. {topic} ({timestamp})")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Topic input
        topic = st.text_input(
            "📝 Temat artykułu:",
            placeholder="np. trądzik na dekolcie, nawilżanie skóry zimą...",
            help="Wpisz temat, o którym chcesz wygenerować artykuł"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_button = st.button(
            "🚀 Generuj artykuł",
            type="primary",
            use_container_width=True,
            disabled=not topic.strip()
        )
    
    # Generate article
    if generate_button and topic.strip():
        with st.spinner("Generuję artykuł..."):
            try:
                article = generate_article(
                    topic.strip(), 
                    target_words, 
                    api_keys, 
                    st.session_state.produkty_db,
                    st.session_state.products_loaded
                )
                
                st.session_state.generated_article = article
                st.session_state.edited_article = article
                
                # Add to history
                timestamp = time.strftime("%H:%M")
                st.session_state.article_history.append((topic.strip(), timestamp))
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Błąd generowania artykułu: {e}")
    
    # Show editor if article exists
    if st.session_state.generated_article:
        st.markdown("---")
        
        # Article stats
        col1, col2, col3, col4 = st.columns(4)
        
        article_to_analyze = st.session_state.edited_article or st.session_state.generated_article
        word_count = len(article_to_analyze.split())
        char_count = len(article_to_analyze)
        h2_count = article_to_analyze.count('## ')
        
        with col1:
            st.metric("📝 Słowa", word_count)
        with col2:
            st.metric("🔤 Znaki", char_count)
        with col3:
            st.metric("📋 Sekcje H2", h2_count)
        with col4:
            if topic and topic.lower() in article_to_analyze.lower():
                st.metric("🎯 SEO", "✅")
            else:
                st.metric("🎯 SEO", "⚠️")
        
        st.markdown("---")
        
        # Editor section
        st.markdown('<div class="editor-header"><h3>📝 Edytor Artykułu</h3></div>', unsafe_allow_html=True)
        
        # Split view: Editor + Preview
        editor_col, preview_col = st.columns([1, 1])
        
        with editor_col:
            st.markdown("**📝 Kod Markdown:**")
            edited_article = st.text_area(
                "Edytuj artykuł:",
                value=st.session_state.edited_article or st.session_state.generated_article,
                height=500,
                help="Edytuj markdown po lewej stronie. Podgląd pojawi się po prawej.",
                label_visibility="collapsed"
            )
        
        with preview_col:
            st.markdown("**👁️ Podgląd na żywo:**")
            with st.container():
                # Custom container with scrolling
                st.markdown("""
                <div style="height: 500px; overflow-y: auto; border: 1px solid #e0e0e0; padding: 1rem; border-radius: 5px; background-color: #fafafa;">
                """, unsafe_allow_html=True)
                
                # Show preview of edited article
                if edited_article.strip():
                    st.markdown(edited_article)
                else:
                    st.info("Wpisz tekst po lewej stronie, aby zobaczyć podgląd")
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Update edited article
        if edited_article != st.session_state.edited_article:
            st.session_state.edited_article = edited_article
            st.rerun()
        
        # Action buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.download_button(
                "📄 Pobierz MD",
                data=edited_article,
                file_name=f"artykul_{topic.replace(' ', '_')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        with col2:
            if st.button("📋 Kopiuj", use_container_width=True):
                st.code(edited_article, language="markdown")
                st.success("Skopiuj kod powyżej!")
        
        with col3:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.edited_article = st.session_state.generated_article
                st.rerun()
        
        with col4:
            if st.button("💾 Zapisz wersję", use_container_width=True):
                if 'saved_versions' not in st.session_state:
                    st.session_state.saved_versions = []
                timestamp = time.strftime("%H:%M:%S")
                st.session_state.saved_versions.append((edited_article, timestamp))
                st.success(f"Wersja zapisana! ({timestamp})")
        
        # Note about live preview
        st.info("💡 **Tip**: Edytuj kod markdown po lewej stronie - podgląd aktualizuje się automatycznie po prawej!")

if __name__ == "__main__":
    main()
