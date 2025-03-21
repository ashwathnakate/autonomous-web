import streamlit as st
import os
import io
import pandas as pd
import seaborn as sns
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import Tool, AgentExecutor, create_tool_calling_agent
from langchain_experimental.utilities import PythonREPL
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from matplotlib import pyplot as plt
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up the search tool and Python REPL tool
search_tool = TavilySearchResults(max_results=1, api_key=os.getenv("TAVILY_API_KEY"))
python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="Executes Python code and returns the result.",
    func=python_repl.run,
)

# Initialize the LLM Agent
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    max_tokens=1024,
    max_retries=2,
    api_key=os.getenv("GROQ_API_KEY"),
)

# Create the chat prompt template that includes chat history
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that can analyze CSV files and generate insights."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

# Define the available tools
tools = [search_tool, repl_tool]

# Create a session ID if it doesn't exist
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{id(st.session_state)}"

# Create the agent
agent = create_tool_calling_agent(llm, tools, prompt_template)

# Initialize memory and agent executor if not in session state
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()

if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = AgentExecutor(
        agent=agent, tools=tools, checkpointer=st.session_state.memory
    )

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "figure_history" not in st.session_state:
    st.session_state.figure_history = []

# Helper Functions
def save_figure_to_session(fig, message_index):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    st.session_state.figure_history.append(
        {"message_index": message_index, "image_data": buf.getvalue()}
    )

def display_figure_from_data(fig_data):
    if "image_data" in fig_data:
        st.image(fig_data["image_data"])

def reset_conversation():
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.figure_history = []
    st.session_state.session_id = f"session_{id(st.session_state)}"
    st.session_state.memory = MemorySaver()
    st.session_state.agent_executor = AgentExecutor(
        agent=agent, tools=tools, checkpointer=st.session_state.memory
    )

# Data Visualization Function
def plot_data(df):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        st.warning("No numeric columns available for visualization.")
        return

    st.subheader("📊 Data Visualizations")

    # Plot pairplot
    with st.expander("📈 Pairplot"):
        try:
            sns.pairplot(df[numeric_cols])
            plt.title("Pairplot of Numeric Features")
            st.pyplot(plt)
            plt.close()
        except Exception as e:
            st.error(f"Error generating pairplot: {e}")

    # Plot correlation heatmap
    with st.expander("🔥 Correlation Heatmap"):
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
            plt.title("Correlation Heatmap")
            st.pyplot(plt)
            plt.close()
        except Exception as e:
            st.error(f"Error generating heatmap: {e}")

    # Plot distribution for numeric columns
    with st.expander("📊 Distribution Plots"):
        for col in numeric_cols:
            try:
                plt.figure(figsize=(6, 4))
                sns.histplot(df[col], kde=True)
                plt.title(f"Distribution of {col}")
                st.pyplot(plt)
                plt.close()
            except Exception as e:
                st.error(f"Error generating distribution plot for {col}: {e}")

# Streamlit App Setup
col1, col2 = st.columns([7, 1])
with col1:
    st.title("💫 Autonomous Web + CSV Analyzer")
    st.markdown("Llama 3.3 🔗 Web Search 🔗 Python Execution 🔗 CSV Analysis")
with col2:
    if st.button("🧹", help="Clear chat history"):
        reset_conversation()
        st.success("Conversation has been reset!")

# File Upload Button
uploaded_file = st.file_uploader("📂 Upload a CSV file for analysis", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 📝 Uploaded CSV File:")
    st.dataframe(df)

    # Perform analysis using the agent
    analysis_prompt = f"Analyze the following CSV data:\n{df.head(10).to_string(index=False)}"
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        full_response = ""
        attempts = 0
        success = False
        while attempts < 2 and not success:
            try:
                for step in st.session_state.agent_executor.stream(
                    {
                        "input": analysis_prompt,
                        "chat_history": st.session_state.chat_history,
                    },
                    {"configurable": {"thread_id": st.session_state.session_id}},
                ):
                    if "output" in step:
                        full_response += step["output"]
                        message_placeholder.markdown(full_response)

                success = True
            except Exception as error:
                if "Failed to call a function" in str(error):
                    attempts += 1
                    full_response = ""
                else:
                    st.error(f"An error occurred: {error}")
                    break

        if not success:
            st.error("Error persists after retries. Please try again.")

        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.chat_history.append(("ai", full_response))

    # Plot data visualizations
    plot_data(df)

# Accept User Input
user_input = st.chat_input("Ask your question:")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append(("human", user_input))

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        attempts = 0
        success = False
        while attempts < 2 and not success:
            try:
                for step in st.session_state.agent_executor.stream(
                    {
                        "input": user_input,
                        "chat_history": st.session_state.chat_history,
                    },
                    {"configurable": {"thread_id": st.session_state.session_id}},
                ):
                    if "output" in step:
                        full_response += step["output"]
                        message_placeholder.markdown(full_response)

                success = True
            except Exception as error:
                attempts += 1
                full_response = ""
                if attempts == 2:
                    st.error(f"An error occurred: {error}")

        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.chat_history.append(("ai", full_response))

# Display conversation history
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
