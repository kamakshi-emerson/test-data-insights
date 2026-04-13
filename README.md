# Data Insights Assistant for Non-Technical Users

## Overview
Data Insights Assistant for Non-Technical Users is a Approachable, clear, patient, informative, supportive Data Insights / Planetary Information agent designed for Text (web_chat, API) interactions.

## Features


## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Run the agent:
```bash
python agent.py
```

## Configuration

The agent uses the following environment variables:
- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key (if using Anthropic)
- `GOOGLE_API_KEY`: Google API key (if using Google)

## Usage

```python
from agent import Data Insights Assistant for Non-Technical UsersAgent

agent = Data Insights Assistant for Non-Technical UsersAgent()
response = await agent.process_message("Hello!")
```

## Domain: Data Insights / Planetary Information
## Personality: Approachable, clear, patient, informative, supportive
## Modality: Text (web_chat, API)