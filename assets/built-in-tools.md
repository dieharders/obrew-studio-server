# Built-In Tools

## Purpose

How to add additional built-in python functions to use as a built-in llm tool.

## Built-in Tool Functions

Built-in tools are Python modules in `backends/tools/built_in_functions/` that agents can use during inference.

### Current Tools

- `calculator.py` - Basic math operations
- `data_transform.py` - Data transformation utilities
- `enhance_text.py` - Text enhancement/formatting
- `retrieval.py` - RAG-based knowledge retrieval

### Adding a New Built-in Tool

1. **Create the tool file** in `backends/tools/built_in_functions/`:

   ```python
   from pydantic import BaseModel, Field

   class Params(BaseModel):
       """Tool description for LLM."""
       param_name: str = Field(..., description="Parameter description")

   async def main(**kwargs: Params) -> str:
       # Tool implementation
       return "result"
   ```

2. **Add to PyInstaller hidden imports** (required for production builds):

   - `package.json` - Add to all build scripts:
     ```
     --hidden-import tools.built_in_functions.your_tool_name
     ```
   - `.github/workflows/release.yml` - Add:
     ```
     --hidden-import "tools.built_in_functions.your_tool_name" \
     ```

3. **Ensure `__init__.py` exists** in both:
   - `backends/tools/__init__.py`
   - `backends/tools/built_in_functions/__init__.py`

### Why Hidden Imports?

PyInstaller only bundles modules it can trace from the main entry point. Since tools are loaded dynamically via `pkgutil.iter_modules()`, PyInstaller can't detect them automatically. The `--hidden-import` flag forces PyInstaller to include them as proper Python modules.
