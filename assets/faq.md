# About Obrew

## What is OpenBrewAi?

We are an organization focused on maker friendly tools. We use those same tools to build products capable of tackling real world problems.

## What is Obrew Studio?

Obrew aims to be something like a game engine that app creators can use to build their own diverse experiences without re-inventing the wheel or staying on top of the latest Ai tech trends. We want to enable businesses to be built on top of Obrew.

## Why Obrew?

In mid 2023 it was difficult to install or even run many popular Ai text-gen tools. None had all the features that were needed without having to install several other plugins/apps.

So I built something that was

- Private
- Easy to install
- Free to use on my hardware
- Ships with batteries included

## Who is this for?

If you are a hacker, Ai Engineer, Prompt Engineer, Python developer, Web Dev, then this tool is for you.

At minimum, all you need is to be able to write instructions in natural language.

## What can be done with Obrew?

- You can run it locally to automate tasks or deploy it on a server.
- Write an Ai app that uses the Obrew API to handle any Ai request.
- You can easily extend its functionality to perform new tasks.
- Use it to experiment with new models, hyper-parameters or prompts.

## Objectives

1. Provide a flexible framework for anyone to solve a domain problem by outlining in natural language how a body of work should be broken down. Obrew holds the opinion that the human user is an effective planner and Ai agents should follow a human direction (even if that means delegating to an AI).

2. This project attempts to make AI more accessible by offering a friendly user experience and the option to run on commodity hardware. We provide creators with everything they need out of the box. This includes RAG for memory, "GPT store" for user created agents, Workbench for user created jobs, a wizard for easy OS level installation, and extensive API.

3. We want creators to think about building useful Ai products and services rather than the infrastructure. We also want to enable creators to build businesses off the back of Obrew.

# What does Obrew offer?

## On-device solution:

Installed without wrangling Python dependencies or Docker containers. Gives you RAG and other AI engineering basics. Provides a transparent, extensive, unified API where others require you to integrate with multiple API's.

## Privacy:

All data, app settings, models are stored locally without sending metrics to a third party for training or tracking. The web UI is actually built using the Obrew API. The web UI and the server code are both open-sourced for anyone to verify.

## Speed:

Users are given several settings to allow for the most optimal experience. The app can also help with determining the correct settings for the device.

## Accessibility:

Allows for people with any device to build AI apps. Users are also provided a friendly UI and low-code methods for creating. Developers and non-developer creators alike can use it.

# Technical Implementation

The backend is written in Python and performs inference locally. The user interacts with the backend via a web UI hosted at https://studio.openbrewai.com.

Text inference is performed using llama-cpp-python. The RAG implementation uses Llama-Index and ChromaDB for the vector database. The server uses FastAPI. The web front-end uses Next.js and TailwindCSS.

Everything else (agents, tools, workflows) is implemented in-house to keep complexity and dependencies low.

## How does this relate to Local AI?

The backend server runs locally on the user's device as an installed native app.

You can choose from curated open-source models to download from HuggingFace and run locally. All data, including conversations, tools, settings are stored locally on device.

It is very much meant to be "hackable" as all data is output into plain text (json or otherwise) and new tools can be added simply by adding files to the appropriate folder on your local disk.

[Back to main README](../README.md)
