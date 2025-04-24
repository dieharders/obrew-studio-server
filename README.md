<h3 align="center">
    Obrew Studio - Your Personal Ai Engine
</h3>

<h3 align="center">
    <img src="assets/images/banner.png" width="360" height="auto" />
    <br>
    <br>
    A general purpose tool for building private Ai desktop apps.
    <br>
    <br>
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" />
    <img src="https://img.shields.io/badge/-Python-000?&logo=Python" />
    <img src="https://img.shields.io/badge/-JavaScript-000?&logo=JavaScript" />
    <img src="https://img.shields.io/badge/-FastAPI-000?&logo=fastapi" />
</h3>

## Table of Contents

- [Supported Models](https://huggingface.co/models?library=gguf&sort=trending)
- [Features](#app-features-roadmap)
- [How to Use](assets/how-to-use.md)
- [Getting Started](assets/getting-started.md)
- [API Documentation](assets/api-docs.md)
- [Build Steps](assets/build-steps.md)
- [Bundling for Release](assets/bundling-for-release.md)
- [Deploy](assets/deploy.md)
- [FAQ](assets/faq.md)

## Introduction

The goal of this project is to be an all-in-one solution for running local Ai that is easy to install, setup and use. It handles all basic building blocks of Ai: inference, memory retrieval (RAG) and storage (vector DB), model file management, and agent/workflow building.

<p align="center">
    <img src="assets/images/obrew-demo.gif" width="360" height="auto" />
</p>

## Description

Obrew Studio is a native app with a GUI that can be configured to allow access from other apps you write or third party services, making it an ideal engine for Ai workloads built on your own tech stack.

<p align="center">
    <img src="assets/images/chat-history.png" height="250" />
    <img src="assets/images/model-explorer.png" height="250" />
</p>

## How It Works

This backend runs a web server that acts as the main gateway to the suite of tools. A WebUI is provided called [Obrew Studio: WebUI](https://studio.openbrewai.com/) to access this server. You can also run in headless mode to access it programmatically via the API.

<p align="center">
    <img src="assets/images/tools.png" height="250" />
    <img src="assets/images/embed-file.png" height="250" />
</p>

To use Ai locally with your private data for free:

- Launch the desktop app and use the GUI to start building
- or navigate your browser to any web app that supports the api
- or connect with a service provider or custom stack that supports the api

<p align="center">
    <img src="assets/images/app-entry.png" height="300" />
    <img src="assets/images/knowledge.png" height="300" />
</p>

## Minimum hardware requirements

- 8GB Disk space
- 8GB Memory

## App Features Roadmap

✅ Run locally<br>
✅ Desktop installers<br>
✅ Save chat history<br>
✅ CPU & GPU support<br>
✅ Windows OS installer<br>
❌ MacOS/Linux installer<br>
❌ Docker config for cloud deployment<br>
❌ Production ready: This project is under active development<br>

## Ai Features Roadmap

✅ Inference: Run open-source LLM models locally<br>
✅ Embeddings: Create vector embeddings from a file/website/media to augment memory<br>
✅ Knowledge Base: Search a vector database with Llama Index to retrieve information<br>
✅ Agents: Customized LLM, can choose or specify tool use<br>
✅ Tool Use: Choose from pre-made or write your own<br>
❌ Workflows: Composable automation of tasks, teams of agents, parallel processing, conditional routing<br>
❌ Monitors: Source citations, observability, logging, time-travel, transparency<br>
❌ Support multi-modal: vision, text, audio, 3d, and beyond<br>
❌ Support multi-device memory sharing (i.e. cluster of macs running single large model)<br>
❌ Support voice-to-text and text-to-speech<br>
❌ Auto Agents: Self-prompting, autonomous agent with access to a sandboxed env<br>

## Supported Model Providers

This is a local first project. The ultimate goal is to support many providers via one API.

✅ [Open-Source](https://huggingface.co)<br>
❌ [Google Gemini](https://gemini.google.com)<br>
❌ [OpenAI](https://openai.com/chatgpt)<br>
❌ [Anthropic](https://www.anthropic.com)<br>
❌ [Mistral AI](https://mistral.ai)<br>
❌ [Groq](https://groq.com)<br>

## Learn More

- Backend: [FastAPI](https://fastapi.tiangolo.com/) - learn about FastAPI features and API.
- Inference: [llama-cpp](https://github.com/ggerganov/llama.cpp) for LLM inference.
- Memory: [Llama-Index](https://github.com/run-llama/llama_index) for data retrieval and [ChromaDB](https://github.com/chroma-core/chroma) for vector database.
- WebUI: Vanilla HTML and [Next.js](https://nextjs.org/) for front-end UI and [Pywebview](https://github.com/r0x0r/pywebview) for rendering the webview.
