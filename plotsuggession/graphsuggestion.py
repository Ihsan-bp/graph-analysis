from langchain_groq import ChatGroq 
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
GROQ_KEY = os.getenv('GROQ_API_KEY')

json_schema = {
    "title": "Plot or graph suggestions",
    "description": "A list of possible plot ideas based on the dataframe.",
    "type": "object",
    "properties": {
        "possible_plots": {
            "type": "array",
            "items": {
                "type": "string",
                "description": "A possible plot or graph idea",
            }
        }
    },
    "required": ["possible_plots"]
}

# Initialize the LLM (LangChain Groq) for generating plot suggestions
llm = ChatGroq(
    model= "llama3-groq-70b-8192-tool-use-preview",
    temperature=0.4,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=GROQ_KEY
)
structured_llm = llm.with_structured_output(json_schema)

def get_plot_suggestions(dataframe_head, df_dtypes, column_names):
    df_head_str = dataframe_head.to_string()
    dtypes = df_dtypes.to_string()
    col_names = ", ".join(column_names)
    
    # Define the system message with instructions for LLM
    prompt = f"""
            You are a data visualization assistant. Your role is to analyze the provided dataset and suggest various plots and graphs that can be generated using Seaborn and Matplotlib.\n
            Below are the first few rows and data types from the dataset:
            Rows:\n{df_head_str}\n
            Data Types:\n{dtypes}\n\n
            Column Names:\n{col_names}\n\n

            Based on this data, suggest relevant plot ideas that can be created using Seaborn and Matplotlib. Consider:
            - Data type analysis (e.g., numerical, categorical)
            - Relationships between columns (e.g., correlation, comparison)
            - Common visualizations such as bar charts, histograms, scatter plots, etc.
            - Ensure the suggestions are implementable using Seaborn and Matplotlib.

            Provide the suggestions in a structured JSON format:
            ```
            {{
                "possible_plots": [
                    "Plot 1: Description of plot (e.g., 'Bar chart of column X')",
                    "Plot 2: Description of plot (e.g., 'Scatter plot of columns X and Y')"
                ]
            }}
            ```
            Only suggest meaningful plot ideas that offer insights into the data, and ensure each suggestion can be visualized with the mentioned tools.
            """
    
    # Call the LLM to get plot suggestions
    order_llm = llm.with_structured_output(json_schema)
    plotidea = order_llm.invoke(prompt)
    return plotidea["possible_plots"]