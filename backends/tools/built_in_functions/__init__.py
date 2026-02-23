# Built-in tool functions
# Static imports for PyInstaller compatibility

from . import (
    calculator,
    data_transform,  # @TODO See if we need this anymore
    enhance_text,
    retrieval,  # @TODO Rename to embeddings_query
    return_array_of_strings,
    return_json,
    return_number,
    return_object,
    return_string,
    return_boolean,
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
    # agent_todo,
    agent_choices,
    agent_choose,
    agent_judge,
    email_grep,
    email_scan,
    email_preview,
    email_read,
    analyze_email,
    widget_email,
    widget_calendar,
    widget_meeting,
    chat_insights,
    sharepoint_read,
    sharepoint_scan,
)

# Registry for tool discovery - maps filename to module
TOOLS = {
    # General tools
    "calculator.py": calculator,
    "data_transform.py": data_transform,
    "enhance_text.py": enhance_text,
    # Embeddings tools
    "retrieval.py": retrieval,
    # Data return tools
    "return_array_of_strings.py": return_array_of_strings,
    "return_boolean.py": return_boolean,
    "return_json.py": return_json,
    "return_number.py": return_number,
    "return_object.py": return_object,
    "return_string.py": return_string,
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
    # Agent tools
    # "agent_todo.py": agent_todo,
    "agent_choices.py": agent_choices,
    "agent_choose.py": agent_choose,
    "agent_judge.py": agent_judge,
    # Widget tools
    "widget_email.py": widget_email,
    "widget_calendar.py": widget_calendar,
    "widget_meeting.py": widget_meeting,
    # Email search tools
    "email_grep.py": email_grep,
    "email_scan.py": email_scan,
    "email_preview.py": email_preview,
    "email_read.py": email_read,
    # Email analysis tools
    "analyze_email.py": analyze_email,
    # Chat analysis tools
    "chat_insights.py": chat_insights,
    # SharePoint tools
    "sharepoint_read.py": sharepoint_read,
    "sharepoint_scan.py": sharepoint_scan,
}
