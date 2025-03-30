import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime
from typing import Dict, List, Any, Optional
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import tempfile

# Import our RAG modules
from src.document_processing import DocumentProcessor
from src.llm_integration_rag import LLMClientRAG

def render_rag_page(crypto_info=None, primary_crypto=None, theme=None):
    """
    Renders the page with RAG (Retrieval-Augmented Generation) functionality
    
    Args:
        crypto_info: Dictionary with cryptocurrency information
        primary_crypto: Primary cryptocurrency for analysis
        theme: Dictionary with theme colors
    """
    st.header("📚 Cryptocurrency Research & Documents")
    
    # Initialize document processor and LLM client
    doc_processor = DocumentProcessor()
    llm_rag_client = LLMClientRAG()
    
    # Create tabbed interface
    rag_tabs = st.tabs(["🔍 Research Assistant", "📄 Document Manager", "📊 Content Analysis"])
    
    # Tab 1: Research Assistant
    with rag_tabs[0]:
        st.subheader("AI Research Assistant")
        
        # Introduction
        st.markdown("""
        Ask questions about cryptocurrencies and get answers based on your documents.
        The assistant uses the documents you've uploaded to provide more accurate and relevant answers.
        """)
        
        # Query input field
        query = st.text_area("Ask a question about crypto:", height=100, 
                              placeholder="Example: What are the main factors affecting Bitcoin price?")
        
        # Search options
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            search_crypto = st.selectbox(
                "Filter by cryptocurrency:", 
                options=["All"] + list(crypto_info.keys()) if crypto_info else ["All"],
                index=0
            )
        
        with col2:
            document_types = ["All Types", "PDF", "HTML", "Text"]
            search_doc_type = st.selectbox("Document type:", options=document_types, index=0)
        
        with col3:
            max_results = st.slider("Max search results:", min_value=3, max_value=10, value=5)
        
        # Search buttons
        col1, col2 = st.columns([1, 1])
        search_button = col1.button("🔍 Search Documents")
        ask_button = col2.button("🤖 Ask Assistant")
        
        # Prepare search filters
        filter_criteria = {}
        if search_crypto != "All":
            filter_criteria['crypto_id'] = search_crypto
            
        if search_doc_type != "All Types":
            doc_type_map = {"PDF": "pdf", "HTML": "html", "Text": "text"}
            filter_criteria['type'] = doc_type_map.get(search_doc_type, None)
        
        # Search documents
        if search_button or ask_button:
            if not query:
                st.warning("Please enter a question or search query.")
            else:
                with st.spinner("Searching documents..."):
                    search_results = doc_processor.search(query, top_k=max_results, filter_criteria=filter_criteria)
                    
                    if not search_results:
                        st.info("No relevant documents found. Try rephrasing your query or uploading more documents.")
                    else:
                        # Display search results
                        st.markdown("### Search Results")
                        
                        for i, result in enumerate(search_results):
                            # Format result
                            metadata = result.get('metadata', {})
                            title = metadata.get('title', 'Untitled Document')
                            source = metadata.get('source', 'Unknown source')
                            doc_type = metadata.get('type', 'text')
                            score = result.get('score', 0.0)
                            
                            with st.expander(f"{i+1}. {title} (Score: {score:.2f})"):
                                st.markdown(f"**Source:** {source}")
                                st.markdown(f"**Type:** {doc_type.upper()}")
                                st.markdown(f"**Relevant excerpt:**")
                                st.markdown(f"```\n{result.get('segment_text', 'No text available')[:500]}...\n```")
                
                # Generate AI response
                if ask_button:
                    if not search_results:
                        st.warning("Cannot generate an answer without relevant documents. Try adjusting your search.")
                    else:
                        with st.spinner("Generating answer..."):
                            # Prepare cryptocurrency context
                            crypto_context = None
                            if search_crypto != "All" and search_crypto in crypto_info:
                                crypto_context = crypto_info[search_crypto]
                            
                            # Generate response
                            response = llm_rag_client.generate_response_with_context(
                                query=query,
                                context_documents=search_results,
                                crypto_info=crypto_context
                            )
                            
                            # Display response
                            st.markdown("### 🤖 AI Response")
                            st.markdown(response.get('answer', 'No answer generated'))
                            
                            # Display sources
                            sources = response.get('sources', [])
                            if sources:
                                with st.expander("Sources"):
                                    for src in sources:
                                        st.markdown(f"- **{src.get('title', src.get('doc_id', 'Unknown'))}**")
                                        if 'source' in src:
                                            st.markdown(f"  Source: {src['source']}")
    
    # Tab 2: Document Manager
    with rag_tabs[1]:
        st.subheader("Document Management")
        
        # Document addition interface
        st.markdown("### Upload Documents")
        
        # File upload form
        upload_col1, upload_col2 = st.columns([2, 1])
        
        with upload_col1:
            uploaded_file = st.file_uploader("Upload PDF or text document", type=["pdf", "txt"])
        
        with upload_col2:
            if primary_crypto and primary_crypto in crypto_info:
                default_crypto = primary_crypto
            else:
                default_crypto = list(crypto_info.keys())[0] if crypto_info else "bitcoin"
                
            doc_crypto = st.selectbox(
                "Related cryptocurrency", 
                options=list(crypto_info.keys()) if crypto_info else ["bitcoin", "ethereum"],
                index=0
            )
        
        # Additional document metadata
        doc_title = st.text_input("Document title (optional)", "")
        doc_source = st.text_input("Source (optional)", "")
        
        # Upload button
        upload_button = st.button("📤 Process Document")
        
        if upload_button and uploaded_file:
            with st.spinner("Processing document..."):
                # Prepare metadata
                metadata = {
                    'crypto_id': doc_crypto,
                    'title': doc_title if doc_title else uploaded_file.name,
                    'upload_date': datetime.now().isoformat(),
                    'source': doc_source if doc_source else "User uploaded"
                }
                
                # Process different file types
                if uploaded_file.name.endswith('.pdf'):
                    # Save temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Process PDF
                    doc_id = doc_processor.process_pdf(tmp_path, metadata)
                    
                    # Remove temporary file
                    os.unlink(tmp_path)
                    
                elif uploaded_file.name.endswith('.txt'):
                    # Process text file
                    text_content = uploaded_file.getvalue().decode('utf-8')
                    doc_id = doc_processor.process_text(text_content, metadata)
                
                if doc_id:
                    st.success(f"Document processed successfully! Document ID: {doc_id}")
                else:
                    st.error("Error processing document. Please try again.")
        
        # URL addition interface
        st.markdown("### Add Web Content")
        
        # URL addition form
        url_col1, url_col2 = st.columns([2, 1])
        
        with url_col1:
            url_input = st.text_input("Enter URL to article or webpage", "")
        
        with url_col2:
            url_crypto = st.selectbox(
                "Related cryptocurrency", 
                options=list(crypto_info.keys()) if crypto_info else ["bitcoin", "ethereum"],
                index=0,
                key="url_crypto"
            )
        
        # Additional metadata
        url_title = st.text_input("Content title (optional)", "")
        
        # URL processing button
        process_url_button = st.button("🌐 Process URL")
        
        if process_url_button and url_input:
            if not url_input.startswith(('http://', 'https://')):
                st.warning("Please enter a valid URL starting with http:// or https://")
            else:
                with st.spinner("Processing web content..."):
                    # Prepare metadata
                    metadata = {
                        'crypto_id': url_crypto,
                        'title': url_title if url_title else "Web content",
                        'upload_date': datetime.now().isoformat(),
                        'source': url_input
                    }
                    
                    # Process URL
                    doc_id = doc_processor.process_html(url_input, metadata)
                    
                    if doc_id:
                        st.success(f"Web content processed successfully! Document ID: {doc_id}")
                    else:
                        st.error("Error processing web content. Please check the URL and try again.")
        
        # Document list
        st.markdown("### Document Library")
        
        # Get list of documents
        documents = doc_processor.list_all_documents()
        
        if not documents:
            st.info("No documents in the library. Upload documents to get started.")
        else:
            # Convert to DataFrame for easier display
            doc_data = []
            for doc in documents:
                metadata = doc.get('metadata', {})
                doc_data.append({
                    'ID': doc.get('id', 'Unknown'),
                    'Title': metadata.get('title', 'Untitled'),
                    'Type': metadata.get('type', 'Unknown').upper(),
                    'Crypto': metadata.get('crypto_id', 'General'),
                    'Date': metadata.get('processed_date', '').split('T')[0] if 'processed_date' in metadata else '',
                    'Source': metadata.get('source', 'Unknown')
                })
            
            doc_df = pd.DataFrame(doc_data)
            
            # Document filters
            filter_col1, filter_col2 = st.columns(2)
            
            with filter_col1:
                filter_type = st.multiselect(
                    "Filter by type",
                    options=sorted(doc_df['Type'].unique().tolist()),
                    default=[]
                )
            
            with filter_col2:
                filter_crypto = st.multiselect(
                    "Filter by cryptocurrency",
                    options=sorted(doc_df['Crypto'].unique().tolist()),
                    default=[]
                )
            
            # Apply filters
            filtered_df = doc_df.copy()
            if filter_type:
                filtered_df = filtered_df[filtered_df['Type'].isin(filter_type)]
            if filter_crypto:
                filtered_df = filtered_df[filtered_df['Crypto'].isin(filter_crypto)]
            
            # Sorting
            filtered_df = filtered_df.sort_values('Date', ascending=False)
            
            # Display document table
            st.dataframe(
                filtered_df[['Title', 'Type', 'Crypto', 'Date', 'Source']],
                use_container_width=True,
                column_config={
                    "Title": st.column_config.TextColumn("Title"),
                    "Type": st.column_config.TextColumn("Type"),
                    "Crypto": st.column_config.TextColumn("Cryptocurrency"),
                    "Date": st.column_config.DateColumn("Date Added"),
                    "Source": st.column_config.TextColumn("Source")
                }
            )
            
            # Document deletion option
            doc_to_delete = st.selectbox(
                "Select document to delete:",
                options=[""] + filtered_df['ID'].tolist(),
                format_func=lambda x: "" if x == "" else f"{filtered_df[filtered_df['ID']==x]['Title'].values[0]} ({x})" if x in filtered_df['ID'].values else x
            )
            
            if doc_to_delete:
                if st.button("❌ Delete Selected Document"):
                    # Add document deletion function here
                    st.warning(f"Document deletion not implemented yet. Would delete document: {doc_to_delete}")
    
    # Tab 3: Content Analysis
    with rag_tabs[2]:
        st.subheader("Document Analysis")
        
        # Get list of documents for analysis
        documents = doc_processor.list_all_documents()
        
        if not documents:
            st.info("No documents available for analysis. Upload documents to get started.")
        else:
            # Prepare list of documents to choose from
            doc_options = []
            for doc in documents:
                metadata = doc.get('metadata', {})
                title = metadata.get('title', 'Untitled')
                doc_id = doc.get('id', '')
                doc_options.append({"label": f"{title} ({doc_id})", "value": doc_id})
            
            # Select document to analyze
            selected_doc_id = st.selectbox(
                "Select document to analyze:",
                options=[opt["value"] for opt in doc_options],
                format_func=lambda x: next((opt["label"] for opt in doc_options if opt["value"] == x), x)
            )
            
            if selected_doc_id:
                # Get full document
                document = doc_processor.get_document_by_id(selected_doc_id)
                
                if document:
                    # Display document information
                    metadata = document.get('metadata', {})
                    doc_title = metadata.get('title', 'Untitled')
                    doc_type = metadata.get('type', 'unknown')
                    doc_crypto = metadata.get('crypto_id', 'General')
                    
                    st.markdown(f"### Analysis of: {doc_title}")
                    
                    # Document information
                    info_col1, info_col2, info_col3 = st.columns(3)
                    
                    with info_col1:
                        st.markdown(f"**Type:** {doc_type.upper()}")
                    with info_col2:
                        st.markdown(f"**Related to:** {doc_crypto}")
                    with info_col3:
                        st.markdown(f"**Date added:** {metadata.get('processed_date', 'Unknown').split('T')[0] if 'processed_date' in metadata else 'Unknown'}")
                    
                    # Document text
                    with st.expander("Document Content"):
                        doc_text = document.get('text', 'No content available')
                        # If text is long, show only a portion
                        if len(doc_text) > 2000:
                            st.markdown(f"{doc_text[:2000]}...")
                            st.markdown(f"*Document is {len(doc_text)} characters long. Showing first 2000 characters.*")
                        else:
                            st.markdown(doc_text)
                    
                    # LLM Analysis
                    st.markdown("### AI Analysis")
                    
                    # Analysis buttons
                    analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
                    
                    with analysis_col1:
                        sentiment_button = st.button("📊 Analyze Sentiment")
                    with analysis_col2:
                        summary_button = st.button("📝 Generate Summary")
                    with analysis_col3:
                        entities_button = st.button("🏷️ Extract Entities")
                    
                    # Sentiment analysis
                    if sentiment_button:
                        with st.spinner("Analyzing sentiment..."):
                            # Extract cryptocurrency symbol if available
                            crypto_symbol = None
                            if doc_crypto in crypto_info:
                                crypto_symbol = crypto_info[doc_crypto].get('symbol', '').upper()
                            
                            # Perform sentiment analysis
                            sentiment_results = llm_rag_client.analyze_document_sentiment(
                                document_text=document.get('text', ''),
                                crypto_symbol=crypto_symbol
                            )
                            
                            # Display results
                            if sentiment_results:
                                # Coloring based on sentiment
                                sentiment = sentiment_results.get('sentiment', 'neutral')
                                sentiment_color = theme['positive'] if sentiment == 'positive' else (
                                    theme['negative'] if sentiment == 'negative' else theme['neutral'])
                                
                                # Results visualization
                                st.markdown(f"#### Sentiment Analysis Results")
                                
                                # Single-line summary
                                st.markdown(f"**Summary:** {sentiment_results.get('summary', 'No summary available')}")
                                
                                # Main sentiment indicator
                                st.markdown(
                                    f"<div style='text-align: center; padding: 20px; margin: 10px 0; "
                                    f"background-color: {sentiment_color}22; border-radius: 10px; border: 1px solid {sentiment_color};'>"
                                    f"<div style='font-size: 24px; font-weight: 600; color: {sentiment_color};'>"
                                    f"Overall Sentiment: {sentiment.upper()}</div>"
                                    f"<div style='font-size: 18px;'>Confidence: {sentiment_results.get('confidence', 0)}%</div>"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )
                                
                                # Key drivers
                                st.markdown("**Key Sentiment Drivers:**")
                                for driver in sentiment_results.get('key_drivers', []):
                                    st.markdown(f"- {driver}")
                                    
                    # Summary generation
                    if summary_button:
                        with st.spinner("Generating summary..."):
                            # Extract cryptocurrency symbol if available
                            crypto_symbol = None
                            if doc_crypto in crypto_info:
                                crypto_symbol = crypto_info[doc_crypto].get('symbol', '').upper()
                            
                            # Generate summary
                            summary = llm_rag_client.summarize_document(
                                document_text=document.get('text', ''),
                                crypto_symbol=crypto_symbol
                            )
                            
                            # Display summary
                            if summary:
                                st.markdown("#### Document Summary")
                                st.markdown(summary)
                    
                    # Entity extraction
                    if entities_button:
                        with st.spinner("Extracting entities..."):
                            # Extract entities
                            entities = llm_rag_client.extract_entities(document.get('text', ''))
                            
                            if entities:
                                st.markdown("#### Named Entities")
                                
                                entity_col1, entity_col2 = st.columns(2)
                                
                                with entity_col1:
                                    # Cryptocurrencies
                                    st.markdown("**Cryptocurrencies:**")
                                    crypto_entities = entities.get('cryptocurrencies', [])
                                    if crypto_entities:
                                        for entity in crypto_entities:
                                            st.markdown(f"- {entity}")
                                    else:
                                        st.markdown("*None detected*")
                                    
                                    # People
                                    st.markdown("**People:**")
                                    people_entities = entities.get('people', [])
                                    if people_entities:
                                        for entity in people_entities:
                                            st.markdown(f"- {entity}")
                                    else:
                                        st.markdown("*None detected*")
                                
                                with entity_col2:
                                    # Organizations
                                    st.markdown("**Organizations:**")
                                    org_entities = entities.get('organizations', [])
                                    if org_entities:
                                        for entity in org_entities:
                                            st.markdown(f"- {entity}")
                                    else:
                                        st.markdown("*None detected*")
                                    
                                    # Technologies
                                    st.markdown("**Technologies:**")
                                    tech_entities = entities.get('technologies', [])
                                    if tech_entities:
                                        for entity in tech_entities:
                                            st.markdown(f"- {entity}")
                                    else:
                                        st.markdown("*None detected*")
                                
                                # Events
                                st.markdown("**Events:**")
                                event_entities = entities.get('events', [])
                                if event_entities:
                                    for entity in event_entities:
                                        st.markdown(f"- {entity}")
                                else:
                                    st.markdown("*None detected*")
                                
                                # Entity visualization
                                if any([len(entities.get(k, [])) > 0 for k in entities.keys()]):
                                    # Prepare data for chart
                                    viz_data = []
                                    for category, entities_list in entities.items():
                                        for entity in entities_list:
                                            viz_data.append({
                                                'category': category.capitalize(),
                                                'entity': entity
                                            })
                                    
                                    if viz_data:
                                        # Create DataFrame
                                        viz_df = pd.DataFrame(viz_data)
                                        
                                        # Prepare colors for categories
                                        color_map = {
                                            'Cryptocurrencies': theme['primary'],
                                            'People': theme['secondary'],
                                            'Organizations': '#FF9800',  # Amber
                                            'Technologies': '#009688',  # Teal
                                            'Events': '#9C27B0'         # Purple
                                        }
                                        
                                        # Create chart
                                        fig = px.treemap(
                                            viz_df,
                                            path=['category', 'entity'],
                                            color='category',
                                            color_discrete_map=color_map,
                                            title='Entity Visualization'
                                        )
                                        
                                        fig.update_layout(
                                            margin=dict(t=30, l=10, r=10, b=10),
                                            height=400
                                        )
                                        
                                        # Display chart
                                        st.plotly_chart(fig, use_container_width=True)