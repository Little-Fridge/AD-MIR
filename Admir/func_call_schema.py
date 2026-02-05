"""
Function Calling Schema Generator
Utilities to convert Python function signatures into OpenAI tool definitions.
"""
import inspect
from typing import get_type_hints, get_origin, get_args, Annotated, Any

def doc(description: str):
    """Annotation tag for describing parameters."""
    return description

def _python_type_to_json_type(py_type):
    """Maps Python types to JSON schema types."""
    if py_type == str:
        return "string"
    if py_type == int:
        return "integer"
    if py_type == float:
        return "number"
    if py_type == bool:
        return "boolean"
    if py_type == list or get_origin(py_type) == list:
        return "array"
    if py_type == dict or get_origin(py_type) == dict:
        return "object"
    return "string"  # Default fallback

def as_json_schema(func: Any) -> dict:
    """
    Converts a Python function into an OpenAI function schema dictionary.
    
    Args:
        func: The function object to analyze.
        
    Returns:
        A dictionary representing the function schema for OpenAI API.
    """
    try:
        type_hints = get_type_hints(func, include_extras=True)
    except Exception:
        # Fallback if type hints fail to resolve
        type_hints = {}
        
    signature = inspect.signature(func)
    params = signature.parameters
    
    properties = {}
    required = []
    
    for param_name, param in params.items():
        # skip self/cls for methods
        if param_name in ("self", "cls"):
            continue
            
        param_type = type_hints.get(param_name, str)
        description = ""
        
        # Extract description from Annotated types, e.g. Annotated[str, doc("...")]
        if get_origin(param_type) is Annotated:
            args = get_args(param_type)
            # The first arg is the actual type
            actual_type = args[0]
            # Look for doc strings in the metadata
            for metadata in args[1:]:
                if isinstance(metadata, str):
                    description = metadata
                    break
            json_type = _python_type_to_json_type(actual_type)
        else:
            json_type = _python_type_to_json_type(param_type)

        prop_schema = {
            "type": json_type,
            "description": description
        }
        
        # Handle Array types specifically
        if json_type == "array":
            # Attempt to infer item type for List[T]
            if get_origin(param_type) is Annotated:
                base_list = get_args(param_type)[0]
                list_args = get_args(base_list)
            else:
                list_args = get_args(param_type)
                
            if list_args:
                item_type = _python_type_to_json_type(list_args[0])
                # Handle List[List[str]] for things like time_ranges
                if item_type == "array": 
                    # Simplified for 2D arrays, assuming string or number inside
                    prop_schema["items"] = {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                else:
                    prop_schema["items"] = {"type": item_type}
            else:
                # Default to string items if unknown
                prop_schema["items"] = {"type": "string"}

        properties[param_name] = prop_schema
        
        # Determine if required (no default value)
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    # Function description from docstring
    func_desc = (func.__doc__ or "").strip()

    return {
        "name": func.__name__,
        "description": func_desc,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }