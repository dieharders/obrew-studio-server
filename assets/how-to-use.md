# How to use the backend service

The primary purpose of this project is to act as a service for whatever Ai enhanced application you wish to build. It will do the heavy-lifting while your app is concerned with the user experience and task flow. It can act as an edge service for a webapp or be run locally to provide output to a file or external service.

## Interaction

As a user you have two ways of interacting with this service:

- via http api endpoints, which gives you programmatic access
- via the graphical interface WebUI which uses the same http api under the hood

## How to use from external device

It is possible to run Obrew Studio (the engine) on one machine and use the WebUI on a different device (like a mobile/tablet) as long as it is on the same network.

1. When starting Obrew Studio, navigate to "Settings" and enable "SSL" (be sure you have already created certificates [see here](deploy.md#start-server-on-localcloud-network-over-https)).
2. Select "Start" to start the server.
3. On a different device, navigate the browser url to `https://studio.openbrew.ai`.
4. Toggle "Advanced Settings" button and enter the address of the server (shown on Obrew Studio start screen) under "Hostname" input.
5. If you created self-signed certificates then the browser will block you from connecting by default, click on "issues connecting" then click on the "click here" link. This will open a new tab that gives you the option to ignore certificates by clicking "Advanced" and selecting "Allow".

## How to create Retrieval Agents with Obrew Studio

These are special kinds of LLM's that are capable of data conversion, summarization and synthesis. They have access to tools that allow them to retrieve information from various sources and provide it as context to the query.

1. Model: Choose a capable LLM: Zephyr 7B, Orca Mini 7B
2. Tools: Assign a previously created Retrieval tool
3. Attention Tab:
   - `Prompt Mode` is overridden to "instruct" when using retrieval tools.
   - `Tool Response`: Use "results" if you want the most direct response to query. Otherwise use "answer" to refine the final response using assigned `Thinking` and `Personality` traits.
   - `Tool Use`: Most of the time you should use "universal" since it works with everything even models not trained for function calling. If you want a possibly faster and more quality experience with tool use then use "native" but you must assign a function calling capable model (currently llama.cpp has spotty support for this and the feature is WIP, not recommended).
4. Thinking:
   - Only used if `Tool Response` is set to "answer"
5. Personality:
   - Only used if `Tool Response` is set to "answer"
6. Performance:
   - `Context Size` should be set to 0 or highest possible to allow for the most context to be used while searching.
7. Memory: Assign a knowledge source. Currently only one collection can be searched by llm.
8. Response:
   - It is best to use a very low `temperature` (0) to keep hallucinations low.
   - Adjust `Max Response Tokens` depending on your needs.

## Adding Custom Tools for Agents

Notes on how to write a new tool (code):

1. One async function must be named `main` per file
2. Functions must be written in Python: `function_name.py`
3. Each function needs a description to help the llm
4. Each function needs a Pydantic class (named "Params") assigned to input args

Notes on how to write `Fields` for each parameter:

- `input_type`="options-sel", "text", "options-multi" (set the input type of this param in WebUI)
- `placeholder`=string (what value to set in WebUI)
- `options_source`="retrieval-template", "installed-models", "memories" (tells the WebUI what menu to display for input)
- `llm_not_required`=boolean (True=llm will not respond with this param, False=llm includes param as part of rresponse)

Where to store the function code:
Dev: In the project's root `backends/tools/built_in_functions`
App: In the `_deps/tools/functions` folder in the installation directory.

- [calculator.py](/backends/tools/built_in_functions/calculator.py) Take a look at the example for reference. This tool takes no user arguments.
- [retrieval.py](/backends/tools/built_in_functions/retrieval.py) For an example of a tool that takes a mix of user arguments and llm arguments.

\*\*Please note you will not be able to import modules that are not already installed by the app.

## API Keys and .env variables

Development: Put your .env file in the base directory of the project.

Installed App: Put your .env file in `_deps` folder in the executable's root directory.

If you do not wish to save a .env file you can also set .env vars from the app start page under "settings".

[Back to README](../README.md)
