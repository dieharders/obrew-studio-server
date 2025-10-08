# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Obrew Studio is a general-purpose tool for building private AI desktop apps. It's an all-in-one solution for running local AI that handles inference, memory retrieval (RAG), vector storage, model file management, and agent/workflow building.

## Architecture

- **Backend**: FastAPI web server (Python) that acts as the main gateway
- **Inference**: llama-cpp for LLM inference with CPU & GPU support
- **Memory**: Llama-Index for data retrieval and ChromaDB for vector database
- **Frontend**: WebUI with vanilla HTML/JS (in `backends/ui/public`) and Next.js
- **Desktop**: Pywebview for rendering the native desktop app

## Key Features

- Run open-source LLM models locally
- Create vector embeddings from files/websites/media
- Knowledge base with vector database search
- Customized agents with tool use
- Chat history and model explorer
- Cross-app API access (headless mode)

## Project Structure

- `/backends/ui/` - Frontend GUI (vanilla HTML/JS/CSS) for settings and WebUI
- `/assets/` - Documentation, images, and guides
- Main server code runs the FastAPI backend

## Development Notes

- This is a local-first project focused on privacy
- Currently supports Windows OS with plans for MacOS/Linux
- Under active development - not production-ready yet
- Minimum requirements: 8GB disk space, 8GB memory
