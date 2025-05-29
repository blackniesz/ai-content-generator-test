import streamlit as st
from generator import show_generator_tab
from analyzer import show_analyzer_tab
from products import load_products_database

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
    .tab-header {
        text-align: center;
        font-size: 1.2rem;
        margin: 1rem 0;
        padding: 0.5rem;
        border-radius: 10px;
        background: linear-gradient(45deg, #f8f9fa, #e9ecef);
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
if 'analyzed_text' not in st.session_state:
    st.session_state.analyzed_text = ""
if 'product_recommendations' not in st.session_state:
    st.session_state.product_recommendations = []

# ========================================
# FUNKCJE API
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

# ========================================
# MAIN APP
# ========================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 AI Content Generator - Dr Ambroziak</h1>', unsafe_allow_html=True)
    
    # Load API keys
    api_keys = load_api_keys()
    if not api_keys:
        st.error("❌ Brak konfiguracji kluczy API. Skontaktuj się z administratorem.")
        st.stop()
    
    # Load products database
    if not st.session_state.products_loaded:
        produkty_db, products_loaded = load_products_database()
        st.session_state.produkty_db = produkty_db
        st.session_state.products_loaded = products_loaded
    
    # Sidebar z globalnym statusem
    with st.sidebar:
        st.title("⚙️ Status systemu")
        
        # Products status
        if st.session_state.products_loaded:
            st.success(f"✅ Baza produktów: {len(st.session_state.produkty_db)} produktów")
        else:
            st.warning("⚠️ Baza produktów: niedostępna")
        
        # API status
        st.info("🔑 API Keys: Skonfigurowane")
    
    # Main tabs
    tab1, tab2 = st.tabs(["📝 Generuj nowy artykuł", "🔍 Analizuj gotowy tekst"])
    
    with tab1:
        st.markdown('<div class="tab-header">🚀 Generator artykułów AI</div>', unsafe_allow_html=True)
        st.markdown("*Wygeneruj kompletny artykuł na dowolny temat z automatycznym dopasowaniem produktów Dr Ambroziak*")
        st.markdown("---")
        show_generator_tab(api_keys, st.session_state.produkty_db, st.session_state.products_loaded)
    
    with tab2:
        st.markdown('<div class="tab-header">🔍 Analizator tekstu i rekomendacje produktów</div>', unsafe_allow_html=True)
        st.markdown("*Przeanalizuj gotowy tekst i otrzymaj inteligentne sugestie miejsc na produkty Dr Ambroziak*")
        st.markdown("---")
        show_analyzer_tab(api_keys, st.session_state.produkty_db, st.session_state.products_loaded)

if __name__ == "__main__":
    main()
