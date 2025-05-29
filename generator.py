import streamlit as st
import anthropic
import requests
from bs4 import BeautifulSoup
import re
import time
from products import find_matching_products

def show_generator_tab(api_keys, produkty_db, products_loaded):
    """Show the article generator tab"""
    
    # Sidebar for generator settings
    with st.sidebar:
        st.markdown("### 📝 Ustawienia generatora")
        
        target_words = st.slider(
            "🎯 Długość artykułu (słowa)",
            min_value=400,
            max_value=5000,
            value=2000,
            step=250,
            help="Docelowa liczba słów w artykule"
        )
        
        st.markdown("---")
        
        # Article history
        if st.session_state.article_history:
            st.subheader("📚 Historia artykułów")
            for i, (topic, timestamp, article) in enumerate(st.session_state.article_history[-5:]):
                if st.button(f"{i+1}. {topic} ({timestamp})", key=f"history_{i}"):
                    # Load article from history
                    st.session_state.generated_article = article
                    st.session_state.edited_article = article
                    st.rerun()
    
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
                    produkty_db,
                    products_loaded
                )
                
                st.session_state.generated_article = article
                st.session_state.edited_article = article
                
                # Add to history
                timestamp = time.strftime("%H:%M")
                st.session_state.article_history.append((topic.strip(), timestamp, article))
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Błąd generowania artykułu: {e}")
    
    # Show hybrid editor if article exists
    if st.session_state.generated_article:
        st.markdown("---")
        show_hybrid_editor(topic if 'topic' in locals() else "")

def show_hybrid_editor(topic):
    """Show the hybrid editor with plain text and markdown preview"""
    
    # Article stats
    col1, col2, col3, col4 = st.columns(4)
    
    article_to_analyze = st.session_state.edited_article or st.session_state.generated_article
    word_count = len(article_to_analyze.split())
    char_count = len(article_to_analyze)
    char_count_without_spaces = len(article_to_analyze.replace(' ', ''))
    h2_count = article_to_analyze.count('## ')
    
    with col1:
        st.metric("📝 Słowa", word_count)
    with col2:
        st.metric("🔤 Znaki", f"{char_count} (ze spacjami)")
    with col3:
        st.metric("🔠 Znaki", f"{char_count_without_spaces} (bez spacji)")
    with col4:
        if topic and topic.lower() in article_to_analyze.lower():
            st.metric("🎯 SEO", "✅")
        else:
            st.metric("🎯 SEO", "⚠️")
    
    # Show more detailed stats
    st.markdown("---")
    
    # Detailed statistics
    detail_col1, detail_col2, detail_col3 = st.columns(3)
    
    with detail_col1:
        st.metric("📊 Sekcje H2", h2_count)
    with detail_col2:
        reading_time = max(1, word_count // 200)  # ~200 words per minute
        st.metric("⏱️ Czas czytania", f"~{reading_time} min")
    with detail_col3:
        if word_count > 0:
            avg_words_per_section = word_count // max(1, h2_count) if h2_count > 0 else word_count
            st.metric("📝 Średnio słów/sekcja", avg_words_per_section)
    
    st.markdown("---")
    
    # Editor mode selection
    st.subheader("✏️ Edytor artykułu")
    
    editor_mode = st.radio(
        "Wybierz tryb edycji:",
        ["📝 Edytor tekstu (hybrid)", "💻 Edytor markdown (zaawansowany)"],
        horizontal=True,
        help="Tryb hybrid: edytujesz tekst, widzisz markdown. Tryb zaawansowany: edytujesz bezpośrednio markdown."
    )
    
    if editor_mode == "📝 Edytor tekstu (hybrid)":
        show_hybrid_text_editor(article_to_analyze)
    else:
        show_markdown_editor(article_to_analyze)
    
    # Action buttons
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_article = st.session_state.edited_article or st.session_state.generated_article
        st.download_button(
            "📄 Pobierz MD",
            data=current_article,
            file_name=f"artykul_{topic.replace(' ', '_')}.md" if topic else "artykul.md",
            mime="text/markdown",
            use_container_width=True
        )
    
    with col2:
        if st.button("📋 Kopiuj", use_container_width=True):
            st.code(current_article, language="markdown")
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
            st.session_state.saved_versions.append((current_article, timestamp))
            st.success(f"Wersja zapisana! ({timestamp})")

def show_hybrid_text_editor(article_content):
    """Show hybrid editor: plain text input + markdown preview"""
    
    st.markdown('<div class="editor-header"><h4>📝 Hybrid Editor</h4></div>', unsafe_allow_html=True)
    
    # Convert markdown to plain text for editing
    plain_text = markdown_to_plain_text(article_content)
    
    # Split view: Text Editor + Markdown Preview
    editor_col, preview_col = st.columns([1, 1])
    
    with editor_col:
        st.markdown("**✏️ Edytor tekstu:**")
        edited_text = st.text_area(
            "Edytuj jako zwykły tekst:",
            value=plain_text,
            height=500,
            help="Edytuj tekst jak w normalnym edytorze. Formatowanie zostanie automatycznie dodane po prawej stronie.",
            label_visibility="collapsed"
        )
        
        st.info("💡 **Wskazówki:**\n- Nowe linie = nowe akapity\n- Linki będą automatycznie wykryte\n- Listy zaczynaj od '-' lub '•'")
    
    with preview_col:
        st.markdown("**👁️ Podgląd markdown:**")
        # Convert plain text back to markdown
        markdown_content = plain_text_to_markdown(edited_text)
        
        # Scrollable preview container
        preview_container = st.container(height=500)
        with preview_container:
            if edited_text.strip():
                st.markdown(markdown_content)
            else:
                st.info("Wpisz tekst po lewej stronie, aby zobaczyć podgląd")
        
        # Show the generated markdown code
        with st.expander("🔍 Zobacz kod markdown"):
            st.code(markdown_content, language="markdown")
    
    # Update edited article with markdown version
    if edited_text != plain_text:
        st.session_state.edited_article = markdown_content
        st.rerun()

def show_markdown_editor(article_content):
    """Show traditional markdown editor"""
    
    st.markdown('<div class="editor-header"><h4>💻 Markdown Editor</h4></div>', unsafe_allow_html=True)
    
    # Split view: Markdown Editor + Live Preview  
    editor_col, preview_col = st.columns([1, 1])
    
    with editor_col:
        st.markdown("**📝 Kod Markdown:**")
        edited_article = st.text_area(
            "Edytuj artykuł:",
            value=article_content,
            height=500,
            help="Edytuj markdown po lewej stronie. Podgląd pojawi się po prawej.",
            label_visibility="collapsed"
        )
    
    with preview_col:
        st.markdown("**👁️ Podgląd na żywo:**")
        # Scrollable preview container
        preview_container = st.container(height=500)
        with preview_container:
            if edited_article.strip():
                st.markdown(edited_article)
            else:
                st.info("Wpisz tekst po lewej stronie, aby zobaczyć podgląd")
    
    # Update edited article
    if edited_article != st.session_state.edited_article:
        st.session_state.edited_article = edited_article
        st.rerun()
    
    # Note about markdown
    st.info("💡 **Tip**: Edytuj kod markdown po lewej stronie - podgląd aktualizuje się automatycznie po prawej!")

def markdown_to_plain_text(markdown_content):
    """Convert markdown to plain text for hybrid editing"""
    
    # Remove markdown headers but keep the text
    text = re.sub(r'^#{1,6}\s+', '', markdown_content, flags=re.MULTILINE)
    
    # Remove bold/italic markers
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
    
    # Convert markdown lists to simple lists
    text = re.sub(r'^\s*[-*+]\s+', '• ', text, flags=re.MULTILINE)
    
    # Remove extra markdown syntax
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Links
    text = re.sub(r'`([^`]+)`', r'\1', text)              # Code
    
    # Clean up extra whitespace
    text = re.sub(r'\n\n\n+', '\n\n', text)
    
    return text.strip()

def plain_text_to_markdown(plain_text):
    """Convert plain text back to markdown with intelligent formatting"""
    
    lines = plain_text.split('\n')
    markdown_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if not line:
            markdown_lines.append('')
            i += 1
            continue
        
        # Detect headers (lines that look like titles)
        if is_likely_header(line, i, lines):
            # Determine header level
            if i == 0 or (i < 3 and len(line) > 20):
                markdown_lines.append(f'# {line}')
            else:
                markdown_lines.append(f'## {line}')
        
        # Detect lists
        elif line.startswith(('• ', '- ', '* ')):
            markdown_lines.append(f'- {line[2:]}')
        
        # Regular paragraph
        else:
            # Add bold to important phrases
            formatted_line = add_smart_formatting(line)
            markdown_lines.append(formatted_line)
        
        i += 1
    
    return '\n'.join(markdown_lines)

def is_likely_header(line, index, all_lines):
    """Determine if a line is likely a header"""
    
    # Very short lines are probably not headers
    if len(line) < 5:
        return False
    
    # Very long lines are probably not headers  
    if len(line) > 100:
        return False
    
    # Lines ending with punctuation are probably not headers
    if line.endswith(('.', '!', '?', ',')):
        return False
    
    # First line is often a header
    if index == 0:
        return True
    
    # Line after empty line might be header
    if index > 0 and not all_lines[index-1].strip():
        return True
    
    # Contains header-like words
    header_words = ['jak', 'dlaczego', 'gdzie', 'kiedy', 'czym', 'przyczyny', 'sposoby', 'metody']
    if any(word in line.lower() for word in header_words):
        return True
    
    return False

def add_smart_formatting(text):
    """Add smart formatting like bold to important phrases"""
    
    # Bold important phrases
    important_phrases = [
        r'\b(ważne|istotne|kluczowe|najważniejsze|pamiętaj|uwaga)\b',
        r'\b(pierwsz[aey]|główn[aey]|podstawow[aey])\b',
        r'\b(skuteczn[aey]|najlepsz[aey]|idealn[aey])\b'
    ]
    
    for phrase_pattern in important_phrases:
        text = re.sub(phrase_pattern, r'**\1**', text, flags=re.IGNORECASE)
    
    return text

# ========================================
# ARTICLE GENERATION FUNCTIONS
# ========================================

def generate_article(topic, target_words, api_keys, produkty_db, products_loaded):
    """Main article generation workflow"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Competition analysis
    status_text.text("🔍 Analizuję konkurencję...")
    progress_bar.progress(0.1)
    
    competition = search_competition(topic, api_keys['google_api'], api_keys['google_cx'])
    
    # Step 2: Information gathering
    status_text.text("📚 Zbiera informacje...")
    progress_bar.progress(0.25)
    
    search_results = search_information(topic, api_keys['google_api'], api_keys['google_cx'])
    all_content = ""
    
    for i, result in enumerate(search_results):
        content = extract_page_content(result['url'], result['title'], result['snippet'])
        all_content += f"\n--- Źródło {i+1}: {result['title']} ---\n{content}\n"
    
    # Step 3: Fact analysis
    status_text.text("🤖 Analizuję fakty przez Claude...")
    progress_bar.progress(0.4)
    
    facts = analyze_facts(all_content, topic, api_keys['anthropic'])
    
    # Step 4: Create outline
    status_text.text("📋 Tworzę konspekt...")
    progress_bar.progress(0.55)
    
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
        progress_bar.progress(0.6 + (0.3 * i / len(section_titles)))
        
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
    progress_bar.progress(0.95)
    
    final_article = f"# {title}\n\n"
    if intro:
        final_article += f"{intro}\n\n"
    final_article += "\n\n".join(sections)
    
    # Add Dr Ambroziak promotion if needed
    if "dr ambroziak" not in final_article.lower() and products_loaded:
        final_article += f"\n\n---\n\n**Profesjonalna pielęgnacja skóry** to podstawa zdrowia i piękna. Jeśli szukasz skutecznych kosmetyków opartych na najnowszych osiągnięciach dermatologii, sprawdź [ofertę Dr Ambroziak Laboratorium](https://drambroziak.com) - produkty stworzone przez ekspertów z ponad 20-letnim doświadczeniem."
    
    progress_bar.progress(1.0)
    status_text.text("✅ Artykuł gotowy!")
    
    return final_article

# Helper functions for article generation
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
            model="claude-3-7-sonnet-20250219",
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
    2. Krótki wstęp (3-5 zdań) - zajawka bez użycia zwrotów typu "w tym artykule", z naturalnym hookiem.
    3. 5-7 śródtytułów (H2) - zróżnicowanych, zawierających long-tail keywords
    4. Dla każdego H2 krótki opis treści (1-2 zdania)
    5. Cel: około {target_words} słów w całym artykule
    6. Styl: przyjazny, lifestyle'owy, praktyczny, user-friendly. Wciel się w rolę pomocnego przewodnika po temacie.
    7. Wyeliminuj wszelkie słowa i zwroty, które mogą świadczyć o AI, takie jak: kluczowy, innowacyjny, holistyczny, nowatorski itp. 

    Format markdown:
    # Główny tytuł H1
    
    [Wstęp 3-5 zdań bez "w tym artykule"]
    
    ## 1. Pierwszy śródtytuł H2
    Opis: Co będzie w tej sekcji...
    
    ## 2. Drugi śródtytuł H2  
    Opis: Co będzie w tej sekcji...
    
    STRUKTURA POWINNA BYĆ UROZMAICONA - niektóre sekcje mogą zawierać listy punktowane, ale tylko tam, gdzie to zasadne.
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
    - Wyeliminuj wszelkie słowa i zwroty, które mogą świadczyć o AI, takie jak: kluczowy, innowacyjny, holistyczny, nowatorski itp.
    
    Zwróć TYLKO treść sekcji, bez dodatkowych komentarzy.
    """
    
    try:
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        st.error(f"Błąd pisania sekcji: {e}")
        return ""
