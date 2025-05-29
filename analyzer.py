import streamlit as st
import anthropic
import docx
import io
from products import analyze_text_for_products, generate_product_suggestion

def show_analyzer_tab(api_keys, produkty_db, products_loaded):
    """Show the text analyzer tab"""
    
    # Input method selection
    st.subheader("📄 Wybierz sposób dodania tekstu")
    
    input_method = st.radio(
        "Jak chcesz dodać tekst do analizy?",
        ["📝 Wklej tekst", "📁 Prześlij plik"],
        horizontal=True
    )
    
    text_to_analyze = ""
    
    if input_method == "📝 Wklej tekst":
        st.markdown("### Wklej swój tekst:")
        text_to_analyze = st.text_area(
            "Artykuł do analizy:",
            height=300,
            placeholder="Wklej tutaj swój artykuł...\n\nMożesz wkleić tekst w dowolnym formacie - zwykły tekst, markdown, czy skopiowany z Word.",
            help="Wklej cały artykuł lub fragment, który chcesz przeanalizować pod kątem miejsc na produkty Dr Ambroziak"
        )
        
    else:  # File upload
        st.markdown("### Prześlij plik:")
        uploaded_file = st.file_uploader(
            "Wybierz plik",
            type=['txt', 'md', 'docx'],
            help="Obsługiwane formaty: .txt, .md, .docx"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.type == "text/plain":
                    text_to_analyze = str(uploaded_file.read(), "utf-8")
                elif uploaded_file.type == "text/markdown":
                    text_to_analyze = str(uploaded_file.read(), "utf-8")
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    doc = docx.Document(io.BytesIO(uploaded_file.read()))
                    text_to_analyze = "\n\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
                
                if text_to_analyze:
                    st.success(f"✅ Plik wczytany! Długość: {len(text_to_analyze)} znaków")
                    
                    # Preview
                    with st.expander("👀 Podgląd wczytanego tekstu"):
                        st.text_area("Treść:", text_to_analyze[:1000] + "..." if len(text_to_analyze) > 1000 else text_to_analyze, height=200, disabled=True)
                        
            except Exception as e:
                st.error(f"❌ Błąd wczytywania pliku: {e}")
    
    # Analysis section
    if text_to_analyze.strip():
        st.markdown("---")
        
        # Text stats
        col1, col2, col3, col4 = st.columns(4)
        word_count = len(text_to_analyze.split())
        char_count = len(text_to_analyze)
        paragraph_count = len([p for p in text_to_analyze.split('\n\n') if p.strip()])
        
        with col1:
            st.metric("📝 Słowa", word_count)
        with col2:
            st.metric("🔤 Znaki", char_count)
        with col3:
            st.metric("📋 Akapity", paragraph_count)
        with col4:
            if products_loaded:
                st.metric("🔍 Status", "✅ Gotowe")
            else:
                st.metric("🔍 Status", "⚠️ Brak bazy")
        
        # Analyze button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button(
                "🔍 Analizuj tekst i znajdź miejsca na produkty",
                type="primary",
                use_container_width=True,
                disabled=not products_loaded
            )
        
        if not products_loaded:
            st.warning("⚠️ Analiza produktów nie jest dostępna - brak bazy danych produktów.")
        
        # Analysis results
        if analyze_button and products_loaded:
            with st.spinner("🔍 Analizuję tekst i szukam miejsc na produkty..."):
                try:
                    # Analyze text for product opportunities
                    recommendations = analyze_text_for_products(
                        text_to_analyze, 
                        produkty_db, 
                        api_keys['openai']
                    )
                    
                    # Store in session state
                    st.session_state.analyzed_text = text_to_analyze
                    st.session_state.product_recommendations = recommendations
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Błąd analizy: {e}")
    
    # Show recommendations if available
    if st.session_state.product_recommendations:
        st.markdown("---")
        st.markdown("## 💡 Rekomendacje produktów")
        
        if len(st.session_state.product_recommendations) == 0:
            st.info("🤔 Nie znaleziono oczywistych miejsc na produkty Dr Ambroziak w tym tekście.")
        else:
            st.success(f"✅ Znaleziono {len(st.session_state.product_recommendations)} możliwości!")
            
            # Show recommendations
            for i, rec in enumerate(st.session_state.product_recommendations):
                with st.expander(f"💡 Rekomendacja {i+1}: {rec['product']['nazwa']} (podobieństwo: {rec['product']['similarity']:.1%})"):
                    
                    # Product info
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**📍 Miejsce:** Akapit {rec['paragraph_index']}")
                        st.markdown(f"**📝 Fragment tekstu:**")
                        st.text_area("", rec['paragraph_text'], height=100, disabled=True, key=f"fragment_{i}")
                        
                    with col2:
                        st.markdown(f"**🛍️ Produkt:** {rec['product']['nazwa']}")
                        st.markdown(f"**🎯 Zastosowanie:** {rec['product']['zastosowanie']}")
                        st.markdown(f"**💰 Cena:** {rec['product'].get('cena', 'N/A')}")
                        st.markdown(f"**🔗 Link:** [Zobacz produkt]({rec['product']['url']})")
                    
                    # Generate suggestion
                    col1, col2 = st.columns([3, 1])
                    with col2:
                        if st.button(f"✨ Generuj sugestię", key=f"gen_sugg_{i}"):
                            with st.spinner("Generuję sugestię..."):
                                try:
                                    anthropic_client = anthropic.Anthropic(api_key=api_keys['anthropic'])
                                    suggestion = generate_product_suggestion(
                                        rec['paragraph_text'],
                                        rec['product'],
                                        rec['suggestion_type'],
                                        anthropic_client
                                    )
                                    st.session_state[f"suggestion_{i}"] = suggestion
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Błąd generowania sugestii: {e}")
                    
                    # Show suggestion if generated
                    if f"suggestion_{i}" in st.session_state:
                        st.markdown("**💬 Sugerowane zdanie do wstawienia:**")
                        suggestion_text = st.text_area(
                            "Możesz edytować sugestię:",
                            st.session_state[f"suggestion_{i}"],
                            height=80,
                            key=f"editable_sugg_{i}"
                        )
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("📋 Skopiuj sugestię", key=f"copy_sugg_{i}"):
                                st.code(suggestion_text)
                                st.success("Skopiuj kod powyżej!")
                        with col2:
                            if st.button("💾 Zapisz zmiany", key=f"save_sugg_{i}"):
                                st.session_state[f"suggestion_{i}"] = suggestion_text
                                st.success("Zapisano!")
            
            # Export recommendations
            st.markdown("---")
            st.markdown("### 📤 Eksport rekomendacji")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Create summary text
                summary_text = create_recommendations_summary(st.session_state.product_recommendations)
                st.download_button(
                    "📄 Pobierz podsumowanie",
                    data=summary_text,
                    file_name="rekomendacje_produktow.txt",
                    mime="text/plain"
                )
            
            with col2:
                if st.button("🔄 Analizuj ponownie"):
                    st.session_state.product_recommendations = []
                    st.session_state.analyzed_text = ""
                    st.rerun()
            
            with col3:
                if st.button("🗑️ Wyczyść wyniki"):
                    st.session_state.product_recommendations = []
                    st.session_state.analyzed_text = ""
                    # Clear all suggestions
                    keys_to_remove = [key for key in st.session_state.keys() if key.startswith('suggestion_')]
                    for key in keys_to_remove:
                        del st.session_state[key]
                    st.rerun()

def create_recommendations_summary(recommendations):
    """Create a summary text of all recommendations"""
    summary = "REKOMENDACJE PRODUKTÓW DR AMBROZIAK\n"
    summary += "=" * 50 + "\n\n"
    
    for i, rec in enumerate(recommendations, 1):
        summary += f"REKOMENDACJA {i}:\n"
        summary += f"Produkt: {rec['product']['nazwa']}\n"
        summary += f"Podobieństwo: {rec['product']['similarity']:.1%}\n"
        summary += f"Miejsce: Akapit {rec['paragraph_index']}\n"
        summary += f"Fragment: {rec['paragraph_text']}\n"
        summary += f"Link: {rec['product']['url']}\n"
        
        if f"suggestion_{i-1}" in st.session_state:
            summary += f"Sugestia: {st.session_state[f'suggestion_{i-1}']}\n"
        
        summary += "\n" + "-" * 30 + "\n\n"
    
    summary += f"Wygenerowano: {len(recommendations)} rekomendacji\n"
    summary += "Generator: AI Content Generator - Dr Ambroziak\n"
    
    return summary
