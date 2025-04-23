# How to use the backend service

The primary purpose of this project is to act as a service for whatever Ai enhanced application you wish to build. It will do the heavy-lifting while your app is concerned with the user experience and task flow. It can act as an edge service for a webapp or be run locally to provide output to a file or external service.

## Interaction

As a user you have two ways of interacting with this service:

- via http api endpoints, which gives you programmatic access
- via the graphical interface WebUI which uses the same http api under the hood

## Adding Custom Tools for Agents

Some notes on how to create a new tool:

1. Function name must be named `main`
2. One function per file
3. Functions must be written in Python: `function_name.py`
4. Each function needs a description to help the llm
5. Each function needs a Pydantic class (named "Params") assigned to input args

Where to store the function code:
Dev: In the project's root `backends/tools/built_in_functions`
Prod: In the `tools/functions` folder in the installation directory.

Take a look at the [calculator.py](/backends/tools/built-in/calculator.py) example for reference.

## API Keys and .env variables

Development: Put your .env file in the base directory of the project.

Installed App: Put your .env file in `_deps` folder in the executable's root directory.

If you do not wish to save a .env file you can also set .env vars from the app start page under "settings".

[Back to README](../README.md)
