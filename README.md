# BBanker - Portfolio Optimization Agent

A portfolio optimization application powered by Azure OpenAI that helps analyze and optimize investment portfolios using Modern Portfolio Theory (MPT) and the Efficient Frontier.

## Features

- 📊 Portfolio optimization using mean-variance analysis
- 📈 Efficient Frontier visualization
- 🤖 AI-powered investment advisor using Azure OpenAI
- 📁 Support for CSV and ODS file formats
- 🎯 Risk-return optimization
- 💬 Interactive chat interface

## Prerequisites

- Docker installed on your system
- Azure OpenAI API key

## Quick Start with Docker

### 1. Set up environment variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit the `.env` file and add your Azure OpenAI API key:

```
AZURE_OPENAI_API_KEY=your_api_key_here
```

### 2. Build the Docker image

```bash
docker build -t bbanker .
```

### 3. Run the container

```bash
docker run -p 7860:7860 --env-file .env bbanker
```

### 4. Access the application

Open your browser and navigate to:

```
http://localhost:7860
```

## Usage

1. **Upload Data**: Upload a CSV or ODS file containing historical returns for your assets
2. **Chat with AI**: Ask questions about portfolio optimization, risk, and returns
3. **Optimize Portfolio**: Request portfolio optimization based on your risk preferences
4. **Visualize**: View the Efficient Frontier and optimal portfolio allocations

## File Format

Your input file should contain:
- Columns: Asset names
- Rows: Historical returns (decimal format, e.g., 0.05 for 5%)

Example CSV:
```csv
Asset1,Asset2,Asset3
0.02,0.03,0.01
0.01,-0.01,0.02
-0.01,0.04,0.03
```

## Project Structure

```
BBanker/
├── agent.py              # Basic chat agent
├── portfolio_agent.py    # Main portfolio optimization agent
├── portfolio_tools.py    # Portfolio calculation functions
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker configuration
├── .env.example         # Environment variables template
└── README.md           # This file
```

## Technologies

- **Python 3.11**
- **Gradio** - Web interface
- **Azure OpenAI** - AI-powered assistant
- **NumPy/SciPy** - Portfolio optimization
- **Pandas** - Data handling
- **Matplotlib** - Visualization

## License

MIT
