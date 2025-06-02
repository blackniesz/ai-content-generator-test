import streamlit as st
from generator import show_generator_tab
from analyzer import show_analyzer_tab
from products import load_products_database, get_demo_products # Added get_demo_products
# import sys # No longer needed here, will be imported in main()
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
        # Removed old DEBUG prints and sys.exit()

    # Check if demo products are being used
    demo_products = get_demo_products()
    is_demo_data = False
    if st.session_state.produkty_db and len(st.session_state.produkty_db) == len(demo_products):
        if demo_products and st.session_state.produkty_db and demo_products[0]['nazwa'] == st.session_state.produkty_db[0]['nazwa']:
             is_demo_data = True
    
    if not st.session_state.products_loaded or not st.session_state.produkty_db:
        is_demo_data = True # Treat as demo/fallback if not properly loaded

    # Sidebar z globalnym statusem
    with st.sidebar:
        if is_demo_data:
            st.error("⚠️ UWAGA: System używa obecnie tymczasowej, demonstracyjnej bazy produktów. Główne dane produktów nie mogły zostać załadowane. Funkcjonalność może być ograniczona, a rekomendacje pochodzić z małego zbioru demonstracyjnego.")
        
        st.title("⚙️ Status systemu")
        if st.session_state.products_loaded and not is_demo_data:
            st.success(f"✅ Baza produktów: {len(st.session_state.produkty_db)} produktów")
        elif is_demo_data:
             st.warning(f"⚠️ Baza produktów: Aktywna baza demonstracyjna ({len(st.session_state.produkty_db) if st.session_state.produkty_db else 0} prod.)")
        else:
            st.warning("⚠️ Baza produktów: niedostępna lub błąd ładowania")
        
        st.info("🔑 API Keys: Skonfigurowane") # Assuming API keys are fine
    
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
