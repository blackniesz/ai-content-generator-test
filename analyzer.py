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
                    # Analyze text for product opportunities - NAPRAWIONE: bez API key
                    recommendations = analyze_text_for_products(
                        text_to_analyze, 
                        produkty_db, 
                        None  # Nie potrzebujemy API key do analizy
                    )
                    
                    # Filter recommendations by quality
                    filtered_recommendations = filter_recommendations_by_quality(recommendations, min_threshold=0.4)
                    
                    # Inform user if some recommendations were filtered out
                    if len(recommendations) > len(filtered_recommendations):
                        filtered_count = len(recommendations) - len(filtered_recommendations)
                        st.info(f"ℹ️ Odrzucono {filtered_count} słabo dopasowanych produktów. Pozostawiono tylko te, które dobrze pasują do kontekstu.")
                    
                    # Store in session state
                    st.session_state.analyzed_text = text_to_analyze
                    st.session_state.product_recommendations = filtered_recommendations
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Błąd analizy: {e}")
                    st.error(f"Debug info: {str(e)}")
    
    
    # Show recommendations if available
    if hasattr(st.session_state, 'product_recommendations') and st.session_state.product_recommendations:
        st.markdown("---")
        st.markdown("## 💡 Rekomendacje produktów")
        
        if len(st.session_state.product_recommendations) == 0:
            st.info("🤔 Nie znaleziono oczywistych miejsc na produkty Dr Ambroziak w tym tekście.")
        else:
            st.success(f"✅ Znaleziono {len(st.session_state.product_recommendations)} dobrze dopasowanych możliwości!")
            
            # Show recommendations
            for i, rec in enumerate(st.session_state.product_recommendations):
                # Determine quality of matching
                relevance = rec['product'].get('thematic_relevance', rec['product'].get('similarity', 0))
                
                # Quality indicators
                if relevance >= 0.8:
                    quality_icon = "🎯"
                    quality_text = "Doskonałe dopasowanie"
                elif relevance >= 0.6:
                    quality_icon = "✅"
                    quality_text = "Dobre dopasowanie"  
                else:
                    quality_icon = "⚠️"
                    quality_text = "Słabe dopasowanie - sprawdź ręcznie"
                
                with st.expander(f"{quality_icon} Rekomendacja {i+1}: {rec['product']['nazwa']} ({quality_text})"):
                    
                    # Warning for poor matches
                    if relevance < 0.6:
                        st.warning("⚠️ **Uwaga:** To dopasowanie może być nietrafione. Sprawdź czy produkt rzeczywiście pasuje do kontekstu przed użyciem.")
                    
                    # Product info
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**📍 Miejsce:** Akapit {rec['paragraph_index']}")
                        st.markdown(f"**📝 Fragment tekstu:**")
                        st.text_area("", rec['paragraph_text'], height=100, disabled=True, key=f"fragment_{i}")
                        
                        # Show main topics if available
                        if 'main_topics' in rec and rec['main_topics']:
                            topics_text = ", ".join([topic['topic'] for topic in rec['main_topics']])
                            st.markdown(f"**🏷️ Zidentyfikowane tematy:** {topics_text}")
                        
                    with col2:
                        st.markdown(f"**🛍️ Produkt:** {rec['product']['nazwa']}")
                        st.markdown(f"**🎯 Zastosowanie:** {rec['product']['zastosowanie']}")
                        st.markdown(f"**💰 Cena:** {rec['product'].get('cena', 'N/A')}")
                        st.markdown(f"**🔗 Link:** [Zobacz produkt]({rec['product']['url']})")
                        st.markdown(f"**📊 Dopasowanie:** {relevance:.1%}")
                    
                    # Generate suggestion
                    col1, col2 = st.columns([3, 1])
                    with col2:
                        if st.button(f"✨ Generuj sugestię", key=f"gen_sugg_{i}"):
                            # Check relevance score before generating
                            if relevance < 0.6:
                                st.warning("⚠️ Ten produkt może nie pasować do kontekstu. Sprawdź ręcznie przed użyciem.")
                            
                            with st.spinner("Generuję spójną sugestię..."):
                                try:
                                    anthropic_client = anthropic.Anthropic(api_key=api_keys['anthropic'])
                                    suggestion = generate_product_suggestion(
                                        rec['paragraph_text'],
                                        rec['product'],
                                        rec.get('suggestion_type', 'general'),
                                        anthropic_client
                                    )
                                    
                                    # Check if suggestion is valid
                                    if "PRODUKT_NIE_PASUJE_DO_KONTEKSTU" in suggestion or "nie pasuje do kontekstu" in suggestion.lower():
                                        st.error("❌ Ten produkt nie pasuje do kontekstu tego akapitu.")
                                        st.info("💡 Spróbuj wybrać inny fragment tekstu lub poczekaj na lepsze dopasowania.")
                                    else:
                                        st.session_state[f"suggestion_{i}"] = suggestion
                                        st.rerun()
                                        
                                except Exception as e:
                                    st.error(f"Błąd generowania sugestii: {e}")
                    
                    # Show suggestion if generated
                    if f"suggestion_{i}" in st.session_state:
                        st.markdown("---")
                        st.markdown("**💬 Przeredagowany akapit:**")
                        suggestion_text = st.text_area(
                            "Możesz edytować sugestię:",
                            st.session_state[f"suggestion_{i}"],
                            height=120,
                            key=f"editable_sugg_{i}",
                            help="To jest Twój oryginalny akapit przeredagowany z naturalnie wpleconym produktem"
                        )
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button("📋 Skopiuj tekst", key=f"copy_sugg_{i}"):
                                st.code(suggestion_text)
                                st.success("✅ Skopiuj tekst powyżej!")
                        with col2:
                            if st.button("💾 Zapisz zmiany", key=f"save_sugg_{i}"):
                                st.session_state[f"suggestion_{i}"] = suggestion_text
                                st.success("✅ Zapisano!")
                        with col3:
                            if st.button("🗑️ Usuń sugestię", key=f"delete_sugg_{i}"):
                                if f"suggestion_{i}" in st.session_state:
                                    del st.session_state[f"suggestion_{i}"]
                                st.rerun()
            
            # Export recommendations
            st.markdown("---")
            st.markdown("### 📤 Eksport i zarządzanie")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Create summary text
                summary_text = create_recommendations_summary(st.session_state.product_recommendations)
                st.download_button(
                    "📄 Pobierz podsumowanie",
                    data=summary_text,
                    file_name="rekomendacje_produktow.txt",
                    mime="text/plain",
                    help="Pobierz pełne podsumowanie ze wszystkimi rekomendacjami i sugestiami"
                )
            
            with col2:
                if st.button("🔄 Analizuj ponownie", help="Uruchom analizę od nowa z tymi samymi ustawieniami"):
                    # Clear only recommendations, keep the text
                    st.session_state.product_recommendations = []
                    # Clear all suggestions
                    keys_to_remove = [key for key in st.session_state.keys() if key.startswith('suggestion_')]
                    for key in keys_to_remove:
                        del st.session_state[key]
                    st.rerun()
            
            with col3:
                if st.button("🗑️ Wyczyść wszystko", help="Usuń tekst, rekomendacje i wszystkie sugestie"):
                    st.session_state.product_recommendations = []
                    st.session_state.analyzed_text = ""
                    # Clear all suggestions
                    keys_to_remove = [key for key in st.session_state.keys() if key.startswith('suggestion_')]
                    for key in keys_to_remove:
                        del st.session_state[key]
                    st.rerun()
            
            # Statistics
            if st.session_state.product_recommendations:
                st.markdown("---")
                st.markdown("### 📊 Statystyki analizy")
                
                # Calculate statistics
                total_suggestions = len([key for key in st.session_state.keys() if key.startswith('suggestion_')])
                avg_relevance = sum(rec['product'].get('thematic_relevance', rec['product'].get('similarity', 0)) 
                                 for rec in st.session_state.product_recommendations) / len(st.session_state.product_recommendations)
                
                high_quality = len([rec for rec in st.session_state.product_recommendations 
                                  if rec['product'].get('thematic_relevance', rec['product'].get('similarity', 0)) >= 0.8])
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🎯 Rekomendacje", len(st.session_state.product_recommendations))
                with col2:
                    st.metric("✨ Wygenerowane sugestie", total_suggestions)
                with col3:
                    st.metric("📈 Średnie dopasowanie", f"{avg_relevance:.1%}")
                with col4:
                    st.metric("🏆 Wysokiej jakości", high_quality)

def filter_recommendations_by_quality(recommendations, min_threshold=0.4):
    """Filter out recommendations with very poor thematic relevance"""
    if not recommendations:
        return []
    
    filtered = []
    
    for rec in recommendations:
        relevance = rec['product'].get('thematic_relevance', rec['product'].get('similarity', 0))
        
        if relevance >= min_threshold:
            filtered.append(rec)
        else:
            # Log filtered out recommendations for debugging
            print(f"Filtered out: {rec['product']['nazwa']} (relevance: {relevance:.2f})")
    
    return filtered

def create_recommendations_summary(recommendations):
    """Create a summary text of all recommendations"""
    summary = "REKOMENDACJE PRODUKTÓW DR AMBROZIAK\n"
    summary += "=" * 50 + "\n\n"
    summary += f"Data wygenerowania: {st.session_state.get('analysis_timestamp', 'N/A')}\n"
    summary += f"Liczba rekomendacji: {len(recommendations)}\n\n"
    
    for i, rec in enumerate(recommendations, 1):
        relevance = rec['product'].get('thematic_relevance', rec['product'].get('similarity', 0))
        
        summary += f"REKOMENDACJA {i}:\n"
        summary += f"Produkt: {rec['product']['nazwa']}\n"
        summary += f"Zastosowanie: {rec['product']['zastosowanie']}\n"
        summary += f"Dopasowanie tematyczne: {relevance:.1%}\n"
        summary += f"Miejsce w tekście: Akapit {rec['paragraph_index']}\n"
        summary += f"Cena: {rec['product'].get('cena', 'N/A')}\n"
        summary += f"Link: {rec['product']['url']}\n\n"
        
        summary += f"ORYGINALNY FRAGMENT:\n"
        summary += f'"{rec['paragraph_text']}"\n\n'
        
        # Add main topics if available
        if 'main_topics' in rec and rec['main_topics']:
            topics = ", ".join([topic['topic'] for topic in rec['main_topics']])
            summary += f"Zidentyfikowane tematy: {topics}\n\n"
        
        # Add suggestion if generated
        if f"suggestion_{i-1}" in st.session_state:
            summary += f"PRZEREDAGOWANY AKAPIT:\n"
            summary += f'"{st.session_state[f"suggestion_{i-1}"]}"\n\n'
        
        summary += "-" * 30 + "\n\n"
    
    # Add statistics
    avg_relevance = sum(rec['product'].get('thematic_relevance', rec['product'].get('similarity', 0)) 
                       for rec in recommendations) / len(recommendations) if recommendations else 0
    high_quality_count = len([rec for rec in recommendations 
                             if rec['product'].get('thematic_relevance', rec['product'].get('similarity', 0)) >= 0.8])
    
    summary += "STATYSTYKI:\n"
    summary += f"Średnie dopasowanie tematyczne: {avg_relevance:.1%}\n"
    summary += f"Rekomendacje wysokiej jakości (>80%): {high_quality_count}\n"
    summary += f"Wygenerowanych sugestii: {len([key for key in st.session_state.keys() if key.startswith('suggestion_')])}\n\n"
    
    summary += "=" * 50 + "\n"
    summary += "Generator: AI Content Generator - Dr Ambroziak\n"
    summary += "Wersja: Enhanced with Thematic Analysis\n"
    
    return summary
