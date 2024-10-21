import streamlit as st
import pandas as pd  #data manipulation
import matplotlib.pyplot as plt #for visualization
import seaborn as sns #for visualization
from langchain_experimental.tools import PythonAstREPLTool  # REPL tool for code execution
from langchain.agents import create_tool_calling_agent  # To create agents that call tools
from langchain.tools.render import render_text_description  # For rendering tool descriptions
from langchain.agents import AgentExecutor  # To execute agent logic
from langchain_groq import ChatGroq  # LLM for graph generation and graph suggestion
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder  # To handle prompts
import os  # For environment variable access

from plotsuggession.graphsuggestion import get_plot_suggestions  # Function for plot suggestions
from dotenv import load_dotenv # For loading environment variables
load_dotenv()

# Load API keys
GROQ_KEY = os.getenv('GROQ_API_KEY')

# Streamlit UI Title
st.title("Graph Generation from CSV Data")

# Input file uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])


if uploaded_file is not None:
    # Load and clean data
    df = pd.read_csv(uploaded_file)
    pd.set_option('display.max_columns', None)
    head = df.head(5)
    df_dtypes = df.dtypes
    column_names = df.columns

    # Create the REPL tool
    repl_tool = PythonAstREPLTool(
        locals={"df": df},
        name="repl_tool",
        description=( 
            "A Python REPL tool to generate visualizations using the provided 'df' DataFrame. "
            "The output should be valid Python code that uses libraries such as matplotlib and seaborn "
            "Include all necessary imports, plot configuration, and ensure to use 'plt.show()' to display the plot."
        ),
        verbose=True,
        return_direct=True,        
    )

    # Generate plot suggestions from the cleaned DataFrame head
    if 'plot_suggestions' not in st.session_state:
        st.session_state.plot_suggestions = get_plot_suggestions(head, df_dtypes, column_names)

    # Provide a dropdown to select one of the plot ideas
    selected_plot = st.selectbox("Select a plot idea", st.session_state.plot_suggestions)

    if st.button("Generate Graph"):
        with st.spinner("Generating graph..."):
            # Create the prompt for the selected plot idea
            prompt = selected_plot

            # Prepare the REPL tool for graph generation
            template = f"""
            You are using the `python_repl` tool and are responsible for generating valid Python code to create plots or graphs based on the provided prompt.
            You hold a tool that can generate a graph or plots.
            **Context:**
            1. You have access to the dataset represented by the variable `df`. 
            2. Below is the output from running `df.head()`:
            <df>
            {head}
            </df>
            3. The following are the column names available in the dataset:
            <df>{column_names}</df>
            
            return the tools answer as final answer.
            provide proper code to make graph or plot.
            Generate the graph based on the input prompt and return the `fig` object.
            """
            tools = [repl_tool]
            tool1 = render_text_description(tools)
            tool_names = ", ".join([t.name for t in tools])
            TEMPLATE = template.format(tools=tool1, tool_names=tool_names)
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", TEMPLATE),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
                ("human", prompt)
            ])

            # Initialize LLM for graph generation
            llm_graph = ChatGroq(temperature=0.1, model="llama3-groq-70b-8192-tool-use-preview", api_key=GROQ_KEY)

            # Create agent for graph generation
            agent = create_tool_calling_agent(llm_graph, tools, prompt_template)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
            
            # Invoke the agent to generate the graph code
            graph_code = agent_executor.invoke({"input": prompt})

            # Check if 'intermediate_steps' contains the expected action for code execution
            if graph_code['intermediate_steps']:
                tool_action = graph_code['intermediate_steps'][0][0]
                code_to_execute = tool_action.tool_input['query']
                
                try:
                    # Try to execute the generated code
                    exec(code_to_execute)
                    # Show the plot in Streamlit
                    st.pyplot(plt)
                except Exception as e:
                    # Handle any exceptions that arise from code execution
                    st.error(f"An error occurred while generating the graph: {e}")
            else:
                # If no code is generated, show the output message instead
                st.warning(graph_code['output'])

else:
    st.warning("Please upload a CSV file to generate graphs.")