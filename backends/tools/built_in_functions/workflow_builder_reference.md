# Available Workflow Nodes

## Rules
- Start with "begin", end with "end"
- List nodes in execution order
- Use "text" nodes to provide instructions for "ai-process" nodes
- Place source nodes before the action that needs their data
- Keep workflows between 3-10 nodes

## Available Nodes

Sources (provide data, place before the action that needs them):
- text: Static text or AI instruction
- number: Numeric value
- file: File reference
- data: JSON/structured data
- api: Call an HTTP API
- context: Fetch project data
- context-search: Search across documents, files, or web

Actions (process data, connected in sequence):
- ai-process: AI analysis, summarization, writing, extraction
- data-transform: Transform, filter, or extract specific data
- file-sync: Synchronize files between locations
- email-notify: Send email notification
- calendar: Create/manage calendar events
- meeting: Create online meeting
- compliance-check: Verify against compliance rules
- email-insights: Analyze email content

Exports (save results to files):
- json-output: Save as JSON
- markdown-output: Save as Markdown
- txt-output: Save as text
- csv-output: Save as CSV
- xml-output: Save as XML

Flow Control:
- comparison: Compare two values (true/false result)
- switch: Branch workflow based on condition
- delay: Wait before continuing

Always start with "begin" and end with "end".
Place source nodes before the action that needs their data.
