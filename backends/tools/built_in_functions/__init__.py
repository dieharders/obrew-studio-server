# Built-in tool functions
# Static imports for PyInstaller compatibility

from . import (
    calculator,
    data_transform,  # @TODO See if we need this anymore
    enhance_text,
    retrieval,  # @TODO Rename to embeddings_query
    extract_array_of_strings,
    extract_boolean,
    extract_json,
    extract_number,
    extract_object,
    extract_string,
    file_glob,
    file_grep,
    file_parse,
    file_preview,
    file_read,
    file_scan,
    item_grep,
    item_preview,
    item_query,
    item_read,
    item_scan,
    agent_todo,
    agent_task,
    agent_ask,
    agent_chooser,
    widget_email,
    widget_calendar,
)

# Registry for tool discovery - maps filename to module
TOOLS = {
    # General tools
    "calculator.py": calculator,
    "data_transform.py": data_transform,
    "enhance_text.py": enhance_text,
    # Embeddings tools
    "retrieval.py": retrieval,
    # Data extraction and transform tools
    "extract_array_of_strings.py": extract_array_of_strings,
    "extract_boolean.py": extract_boolean,
    "extract_json.py": extract_json,
    "extract_number.py": extract_number,
    "extract_object.py": extract_object,
    "extract_string.py": extract_string,
    # File-system tools
    "file_glob.py": file_glob,
    "file_grep.py": file_grep,
    "file_parse.py": file_parse,
    "file_preview.py": file_preview,
    "file_read.py": file_read,
    "file_scan.py": file_scan,
    # Structured data tools
    "item_grep.py": item_grep,
    "item_preview.py": item_preview,
    "item_query.py": item_query,
    "item_read.py": item_read,
    "item_scan.py": item_scan,
    # Agent harness tools
    "agent_task.py": agent_task,
    "agent_todo.py": agent_todo,
    "agent_ask.py": agent_ask,
    "agent_chooser.py": agent_chooser,
    # Widget tools
    "widget_email.py": widget_email,
    "widget_calendar.py": widget_calendar,
}
