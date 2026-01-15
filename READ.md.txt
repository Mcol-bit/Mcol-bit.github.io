# RAG Chatbot with Google Gemini & LangChain

**Project Status:** Active | **Role:** Developer | **Tech Stack:** Python, LangChain, Google GenAI

## Overview
This project implements a Retrieval-Augmented Generation (RAG) chatbot capable of answering questions based on specific PDF documents. By leveraging Google's **Gemini-2.5-Flash** model and **LangChain**, the bot provides context-aware answers rather than generic knowledge.

This implementation allows users to upload custom knowledge bases (such as military strategy documents or martial arts history) and query them using natural language.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OQOFWnJPuMxEO_ZXmzJ9P2y0eHH_2n2t)

---

## Key Features
* **LLM Integration:** Utilizes `gemini-2.5-flash` for high-speed, cost-effective inference.
* **Prompt Engineering:** Custom `ChatPromptTemplate` to enforce strict context adherence.
* **Vector Search Ready:** Designed to integrate with Pinecone for scalable document retrieval.
* **Streaming Responses:** Supports token-by-token streaming for a better user experience.

## Code Highlights

### 1. Model Initialization
I utilized the `langchain-google-genai` library to interface with Gemini, setting a temperature of 0.7 to balance creativity with accuracy.

```python
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize Gemini 2.5 Flash
chat = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.7
)