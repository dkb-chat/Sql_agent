from flask import Flask, request, jsonify, render_template
from langchain.chains import create_sql_query_chain
from langchain_google_genai import GoogleGenerativeAI
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError, OperationalError
from langchain_community.utilities import SQLDatabase
import logging

import pandas as pd

from ratelimit import limits, RateLimitException
import time

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Database connection parameters (replace with your actual values or environment variables)
SQL_SERVER = "192.168.1.7"  # Replace with your SQL Server address
SQL_DATABASE = "python_test"  # Replace with your database name
SQL_USERNAME = "user"  # Replace with your database username
SQL_PASSWORD = "Admin@123"  # Replace with your database password
GOOGLE_API_KEY = "AIzaSyD3Xljd9fbo1WrckDYQMrOoK26PLcQl3Hc"  # Replace with your Google API key

# Rate limiting configuration
CALLS_PER_MINUTE = 10

# --- Database Connection ---
def get_db_connection():
    """
    Establishes a connection to the SQL Server database using pyodbc and SQLAlchemy.
    Returns:
        tuple: (SQLDatabase, Engine) if successful, (None, None) otherwise.
    """
    try:
        connection_string = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={SQL_SERVER};"
            f"DATABASE={SQL_DATABASE};"
            f"UID={SQL_USERNAME};"
            f"PWD={SQL_PASSWORD};"
            f"TrustServerCertificate=yes;"  # Use only for testing; consider proper certificate validation in production
        )
        engine = create_engine(f"mssql+pyodbc:///?odbc_connect={connection_string}", pool_pre_ping=True, echo=False)
        db = SQLDatabase(engine, sample_rows_in_table_info=3, include_tables=['pur', 'account', 'item', 'item_group','pur_detail', 'item_ledger','site'])  # Specify tables for the LLM to use
        print("✅ Successfully connected to the database!")
        return db, engine
    except OperationalError as e:
        logging.error(f"Database connection error: {e}")
        print("❌ Failed to connect to the database. Check your connection settings.")
        return None, None  # Return None for both db and engine in case of failure
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        print(f"❌ An unexpected error occurred: {e}")
        return None, None

# --- LLM Initialization ---
def get_llm():
    """
    Initializes the Google Generative AI model.
    Returns:
        GoogleGenerativeAI: LLM instance if successful, None otherwise.
    """
    try:
        llm_instance = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)
        print("✅ Successfully initialized the LLM!")
        return llm_instance
    except Exception as e:
        logging.error(f"Error initializing LLM: {e}")
        print(f"❌ Failed to initialize the LLM: {e}")
        return None

# --- Initialize Database and LLM ---
print("Attempting to connect to the database...")
db, engine = get_db_connection()

print("Attempting to initialize the LLM...")
llm = get_llm()

if db is None or llm is None:
    print("❌ Critical error: Failed to initialize database or LLM. Exiting...")
    exit(1)

# --- Create SQL Query Chain ---
chain = create_sql_query_chain(llm, db)
print("✅ SQL query chain created successfully!")

# --- SQL Execution Function ---
def execute_sql(sql_query, engine):
    """
    Executes the given SQL query against the database.
    Args:
        sql_query (str): The SQL query to execute.
        engine (Engine): SQLAlchemy engine instance.
    Returns:
        list: A list of dictionaries representing the query results, or None if an error occurred.
    """
    try:
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()  # Remove code block markers
        with engine.connect() as connection:
            result = connection.execute(text(sql_query))
            rows = result.fetchall()
            column_names = result.keys()
            if rows:
                df = pd.DataFrame(rows, columns=column_names)
                return df.to_dict(orient='records')  # Convert DataFrame to list of dictionaries
            else:
                return []  # Return an empty list if no results are found
    except Exception as e:
        logging.error(f"Error executing SQL: {e}")
        print(f"❌ Error executing SQL: {e}")
        return None

# --- Rate Limiting ---
@limits(calls=CALLS_PER_MINUTE, period=60)
def query_chain(query):
    """
    Invokes the Langchain SQL query chain, applying rate limiting.
    Args:
        query (str): The user's natural language query.
    Returns:
        str: The generated SQL query.
    Raises:
        Exception: If an error occurs during query chain invocation.
    """
    try:
        response = chain.invoke({"question": query})
        return response
    except Exception as e:
        raise e

# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the main HTML template."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handles user chat requests, generates and executes SQL, and formats the response.
    Returns:
        JSON: A JSON response containing the formatted response or an error message.
    """
    user_input = request.json.get('query')
    if not user_input:
        return jsonify({"error": "No query provided"}), 400

    try:
        sql_query = query_chain(user_input)  # Generate SQL query using Langchain
        print(f"Generated SQL Query: {sql_query}")
        result = execute_sql(sql_query, engine)  # Execute the SQL query
        if result is None:
            return jsonify({"error": "Error executing SQL query"}), 500

        response = {"sql_query": sql_query, "result": result}  # Create a response dictionary
        formatted_response = format_response(response, user_input)  # Format the response, pass user_input

        return jsonify(formatted_response)  # Return the formatted response as JSON

    except RateLimitException:
        return jsonify({"error": "Rate limit exceeded. Please wait a minute and try again."}), 429
    except Exception as e:
        logging.error(f"Error processing chat request: {e}")  # Log the full error
        return jsonify({"error": str(e)}), 500

# --- Response Formatting ---
def format_response(response, user_input):
  """
  Formats the SQL query result into a human-readable response, adding LLM conversational output.
  Args:
      response (dict): A dictionary containing the SQL query and its result.
      user_input (str): The user's original input query.
  Returns:
      dict: A dictionary containing the formatted response.
  """
  sql_query = response.get("sql_query", "")
  result = response.get("result", [])

  if not result:
      return {"formatted_response": f"I'm sorry, but I couldn't find any information related to your question."}

  if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
      # It's likely a table
      table_html = create_table_html(result)
      conversational_output = generate_conversational_output(user_input, sql_query, result)  # Get Conversational output
      return {"formatted_response": f"{table_html}<br><p>{conversational_output}</p>", "type": "table"}

  elif isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
      # Single row, potentially multiple columns
      row = result[0]
      # Format the dictionary into a readable string
      formatted_text = ", ".join([f"{key}: {value}" for key, value in row.items()])
      conversational_output = generate_conversational_output(user_input, sql_query, result) # Get Conversational output
      return {"formatted_response": f"{conversational_output} Here's a bit more detail: {formatted_text}"}

  elif isinstance(result, list) and len(result) > 0:
      # List of single values (e.g., a list of vendor names)
      formatted_text = ", ".join(map(str, result))  # Convert to string and join
      conversational_output = generate_conversational_output(user_input, sql_query, result) # Get Conversational output
      return {"formatted_response": f"{conversational_output} The details include: {formatted_text}"}

  else:
      # Otherwise, format using LLM (or a simpler method)
      formatted_text = str(result)
      conversational_output = generate_conversational_output(user_input, sql_query, result)  # Get Conversational output
      return {"formatted_response": f"{conversational_output} Result {formatted_text}"}

# --- HTML Table Creation ---
def create_table_html(data):
    """
    Generates an HTML table from a list of dictionaries.
    Args:
        data (list): A list of dictionaries representing the table data.
    Returns:
        str: An HTML string representing the table.
    """
    if not data:
        return "<p>No data to display.</p>"

    columns = data[0].keys()
    table_html = "<div class='table-container'><table class='table'>"
    # Table Head
    table_html += "<thead><tr>"
    for column in columns:
        table_html += f"<th>{column}</th>"
    table_html += "</tr></thead>"
    # Table Body
    table_html += "<tbody>"
    for row in data:
        table_html += "<tr>"
        for column in columns:
            table_html += f"<td>{row.get(column, 'N/A')}</td>"  # Use .get() for safe access
        table_html += "</tr>"
    table_html += "</tbody></table></div>"
    return table_html

# --- Conversational Output Generation ---
# --- Conversational Output Generation ---
def generate_conversational_output(user_input, sql_query, result):
    """
    Generates conversational and human-like output based on the query and results, focusing on insights.
    Args:
        user_input (str): The user's original input query.
        sql_query (str): The generated SQL query.
        result (any): The result of the SQL query.
    Returns:
        str: The LLM-generated conversational output.
    """
    try:
        prompt = f"You are a helpful and friendly chatbot assisting a user with their database questions. Respond in a natural, conversational tone, as if you were a human colleague.\n" \
                 f"The user asked: '{user_input}'\n" \
                 f"The SQL query used to find the answer was: '{sql_query}'\n" \
                 f"The database returned this result (presented as a table in the chat interface): '{result}'\n" \
                 f"Based on this information, provide a *brief* summary of the key insights from the data, focusing on the most important takeaways for the user. DO NOT simply repeat the data that is already visible in the table.\n" \
                 f"Instead, provide an overall analysis or highlight any significant trends or patterns. Avoid technical terms. Limit your response to one or two short sentences, and maintain a friendly and helpful tone."
        conversational_output = llm.invoke(prompt)
        return conversational_output
    except Exception as e:
        logging.error(f"Error generating conversational output with LLM: {e}")
        return "I encountered an error while trying to formulate a human-like response.  Please see the data above."

# --- Main ---
if __name__ == '__main__':
    print("Starting Flask application...")
    app.run(debug=True)