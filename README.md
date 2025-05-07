# 🪙 CryptoText Analytics

## 📋 Project Overview

CryptoText Analytics is an advanced web application built with Streamlit that combines cryptocurrency market analysis with state-of-the-art AI tools. The project leverages both historical and real-time market data to deliver:

- **LLM Integration**: Utilizes language models (e.g., OpenAI GPT-3.5-turbo) to analyze news sentiment and generate comprehensive financial analyses.
- **RAG (Retrieval-Augmented Generation) Functionality**: Implements a system that retrieves context-relevant documents (PDF, HTML, text) and generates precise answers to user queries based solely on processed data.
- **Technical Analysis**: Computes key market metrics, technical indicators, and price trends using data sourced from CoinGecko.
- **Interactive Visualizations**: Displays market data, sentiment analysis, and cryptocurrency comparisons through dynamic, interactive charts powered by Plotly.

## 🏷️ Application Layout and Features

The application features several tabs to help you navigate its functionality:

### Price Analysis
- Presents real-time and historical price charts for a selected cryptocurrency.
- Includes technical analysis indicators, daily returns distribution, and key metrics (volatility, Sharpe ratio, etc.).

  ![image](https://github.com/user-attachments/assets/6b60f956-9dfa-48c0-a6c0-91586bcd0948)


### Crypto Comparison
- Enables side-by-side comparisons of multiple cryptocurrencies.
- Normalizes data to show relative performance on a single chart.

  ![image](https://github.com/user-attachments/assets/4722d0a2-efcf-459f-850a-bfd8da3ae585)


### Event Calendar
- Displays upcoming events and project milestones in the crypto space.
- Fetches data from CoinGecko (or generates mock events) for a consolidated timeline.

![image](https://github.com/user-attachments/assets/9b72f6b5-675a-46a6-87d6-13dec33c038e)


### AI Insights
- Uses advanced language models to provide deeper sentiment analysis and generate financial insights.
- Summaries of recent news, risk assessments, and custom AI-driven reports.

![image](https://github.com/user-attachments/assets/7a54df06-c763-4c66-b70b-2b3473ecae3c)

- Asking questions about cryptocurrency.

![image](https://github.com/user-attachments/assets/8677683c-5bc8-4d02-a3d2-ad0bb2f5c17b)

### Research & Documents
- Allows you to upload and manage documents (PDF, HTML, text).
- Leverages RAG to retrieve relevant content and answer questions using AI, based on your uploaded materials.

![image](https://github.com/user-attachments/assets/6d7d032b-afe9-4756-9441-8ac5b1677a01)
  
## 🗂️ Repository Structure

```
CryptoText-Analytics/
├── data/                       # Directory for data (CSV, JSON, processed files, etc.)
│   ├── crypto_data/            # Historical and real-time cryptocurrency data
│   ├── crypto_events/          # Information on market and project events
│   └── documents/              # Processed documents (PDF, HTML, text)
├── notebooks/                  # Jupyter Notebooks for analysis or usage examples
│   └── 01_data_loading.ipynb
├── src/                        # Main application source code (Python)
│   ├── __init__.py
│   ├── analysis.py             # Data analysis and metric calculations
│   ├── data_loader.py          # Fetches data from the CoinGecko API and generates demo data
│   ├── document_processing.py  # Processes documents for the RAG module
│   ├── event_calendar.py       # Retrieves and simulates cryptocurrency events
│   ├── llm_integration.py      # Integrates LLM for sentiment analysis and report generation
│   ├── llm_integration_rag.py  # Extended LLM integration with RAG functionality
│   ├── rag_page.py             # User interface for RAG features (document search, AI Assistant)
│   └── visualization.py        # Data visualizations (charts, dashboards)
├── venv/                       # Virtual environment (optional – typically ignored by Git)
├── .env                        # Environment variables file (API keys, etc.)
├── app.py                      # Main application file (Streamlit entry point)
├── requirements.txt            # List of project dependencies
└── README.md                   # Project documentation
```

## 🚀 How to Run the Project

1. Clone the repository:
```bash
git clone https://github.com/your-username/CryptoText-Analytics.git
cd CryptoText-Analytics
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Configure the environment:
   - Create a `.env` file in the root directory and add your API keys. For example:
```
OPENAI_API_KEY=your_openai_api_key
```

4. Run the application:
```bash
streamlit run app.py
```

## 💡 Future Developments and Improvements

### Enhancing LLM & RAG Functionality
Plan to further develop the RAG module by integrating additional document sources, improving search precision, and optimizing prompts for the language models.

### Advanced Sentiment Analysis
Intent to enhance sentiment analysis using the latest language models and adaptive strategies to better assess market context.

### Expanding Data Sources
Aim to integrate new APIs and alternative data sources to provide a more comprehensive view of the cryptocurrency market.

### Improving the User Interface
Plan to further refine the UI/UX by expanding dashboard features and introducing additional interactive functionalities.

## 🤝 Contributing

Contributions are welcome! If you have ideas for improvements, notice any bugs, or would like to add new features, please open an issue or submit a pull request. Your suggestions are greatly appreciated.

## 🔒 License

All rights are reserved. No formal license has been chosen for this project at this time.

---

Thank you for your interest in CryptoText Analytics. This application aims to enhance your understanding of the cryptocurrency market and inspire further innovations in data analysis and artificial intelligence.
