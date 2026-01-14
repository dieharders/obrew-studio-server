<h3 align="center">
    OpenBrew - Your Personal Ai Engine
</h3>

<h3 align="center">
    <img src="assets/images/banner.png" width="auto" height="320" />
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
- [Quick Start Guide](assets/quick-start.md)
- [Getting Started](assets/getting-started.md)
- [How to Connect Apps](assets/how-to-connect-apps.md)
- [API Documentation](assets/api-docs.md)
- [Build Steps](assets/build-steps.md)
- [Bundling for Release](assets/bundling-for-release.md)
- [Deploy](assets/deploy.md)
- [FAQ](assets/faq.md)

## Introduction

The goal of this project is to be an all-in-one solution for running local Ai that is easy to install, setup and use. It handles all basic building blocks of Ai: inference, memory retrieval (RAG) and storage (vector DB), model file management, and agent/workflow building.

https://github.com/user-attachments/assets/1aadbc35-b1bd-4489-996d-aea31f6171d6

## Description

OpenBrew is a native app with a GUI that can be configured to allow access from other apps you write or third party services, making it an ideal engine for Ai workloads built on your own tech stack.

<p align="center">
    <img src="assets/images/home.PNG" height="300" />
    <img src="assets/images/startup.PNG" height="300" />
</p>

## How It Works

This backend runs a web server that acts as the main gateway to the suite of tools. A WebUI is provided called [OpenBrew: WebUI](https://studio.openbrew.ai/) to access this server. You can also run in headless mode to access it programmatically via the API.

<p align="center">
    <img src="assets/images/obrew-apps.PNG" height="300" />
    <img src="assets/images/app-qr.PNG" height="300" />
</p>

### To use Ai locally with your private data for free:

- Launch the desktop app and choose an app to start using
- or navigate your browser to any web app that supports the API
- or connect with a service provider or custom stack that supports the OpenBrew API

## Minimum Hardware Requirements

- 8GB Disk space
- 4GB Memory

## App Features Roadmap

✅ Run locally<br>
✅ Windows OS installer<br>
✅ MacOS installer<br>
✅ Save chat history<br>
✅ CPU & GPU support<br>
❌ Linux installer<br>
❌ Production ready: This project is under active development<br>

## Ai Features Roadmap

✅ Inference: Run open-source LLM models locally<br>
✅ Embeddings: Create vector embeddings from a file/website/media to augment memory<br>
✅ Knowledge Base: Search a vector database with Llama Index to retrieve information<br>
✅ Agents: Customized LLM, can choose or specify tool use<br>
✅ Tool Use: Choose from pre-made or write your own<br>
✅ Multi-modal:

- ✅ image
- ✅ text
- ❌ video
- ❌ audio
- ❌ 3d
  <br>

❌ Observability: Source citations, logging, tracing<br>
❌ Cached Context & Extended Context<br>
❌ Voice-to-Text and Text-to-Speech<br>

<!-- ❌ Multi-device memory sharing (i.e. cluster of macs running single large model)<br> -->

## Supported Model Providers

This is a local first project. The ultimate goal is to support many providers via one API.

✅ [Open-Source (GGUF format)](https://huggingface.co)<br>
❌ [Google Gemini](https://gemini.google.com)<br>
❌ [OpenAI](https://openai.com/chatgpt)<br>
❌ [Anthropic](https://www.anthropic.com)<br>
❌ [Mistral AI](https://mistral.ai)<br>
❌ [Groq](https://groq.com)<br>

## Learn More

- Backend: [FastAPI](https://fastapi.tiangolo.com/) - learn about FastAPI features and API.
- Inference: [llama-cpp](https://github.com/ggerganov/llama.cpp) for LLM inference.
- Memory: [ChromaDB](https://github.com/chroma-core/chroma) for vector database.
- WebUI: React for front-end UI and [Pywebview](https://github.com/r0x0r/pywebview) for rendering the webview.
