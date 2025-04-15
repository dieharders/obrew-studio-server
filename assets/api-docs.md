# API Overview

This project deploys several servers/processes (databases, inference, etc.) exposed using the `/v1` endpoint. The goal is to separate all OS level logic and processing from the client apps. This can make deploying new apps and swapping out functionality easier.

A complete list of endpoint documentation can be found at [http://localhost:8000/docs](http://localhost:8000/docs) after Obrew Server is started.

## Client api library

There is currently a javascript library under development and being used by [Obrew Studio WebUI](https://github.com/dieharders/brain-dump). Once the project becomes stable, it will be broken out into its own module and repo. Stay tuned.

[Back to README](../README.md)
