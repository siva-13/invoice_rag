# from pydoc import text
# from fastapi import APIRouter, Depends, HTTPException,status
# from sqlalchemy.orm import Session
# from database.database import get_db
# from services.auth import get_current_user
# from services.openai_services import generate_sql_query, synthesize_response
# from pydantic import BaseModel
# from database.models import User
# from config import DEVICE, process_pool, PDF_IMAGE_DIR, UPLOAD_DIR,MAX_WORKERS,BATCH_SIZE,RATE_LIMIT_REQUESTS,RATE_LIMIT_WINDOW,MAX_CONCURRENT_REQUESTS,API_SEMAPHORE,client,api_key
# import asyncio



# router = APIRouter(tags=["Query"])

# class QueryRequest(BaseModel):
#     query: str

# @router.post("/query-invoices")
# async def query_invoices(
#     query: QueryRequest,
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """
#     Process natural language queries about invoices and return results
#     Example queries:
#     - "What's my total invoice amount for last month?"
#     - "Show me all invoices from seller X"
#     - "What's the average invoice amount?"
#     """
#     try:
#         print(query)
#         # Generate SQL query from natural language
#         sql_query = await generate_sql_query(query,current_user.unique_id)
        
#         # Add user_id filter if not present (security measure)
#         if "WHERE" not in sql_query.upper():
#             sql_query = f"{sql_query} WHERE user_id = {current_user.unique_id}"
#         elif "user_id" not in sql_query:
#             sql_query = sql_query.replace("WHERE", f"WHERE user_id = {current_user.unique_id} AND")
            
#         # Execute query with parameters
#         result = await asyncio.get_running_loop().run_in_executor(
#             None,
#             lambda: db.execute(
#                 text(sql_query),
#                 {"user_id": current_user.unique_id}
#             ).fetchall()
#         )
#         print(f"sql response: {result}")
#         # Convert result to list of dictionaries
#         results = [
#             {column: value for column, value in zip(row._mapping.keys(), row._mapping.values())}
#             for row in result
#         ]
        
#         # Synthesize natural language response
#         explanation = await synthesize_response(query, results, client)
        
#         return {
#             "query": query,
#             "sql_query": sql_query,
#             "results": results,
#             "explanation": explanation
#         }
        
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error processing query: {str(e)}"
#         )


import asyncio
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from database.database import get_db
from services.auth import get_current_user
from database.models import User
from sqlalchemy import text
from services.openai_services import encode_image, process_single_image


from config import client, api_key  # Assuming you have the API key stored here
import requests  # For making API calls to Groq's endpoint

router = APIRouter(tags=["Query"])

class QueryRequest(BaseModel):
    query: str

from sqlalchemy.ext.asyncio import AsyncSession

@router.post("/query-invoices")
async def query_invoices(
    query: QueryRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    try:
        sql_query = await generate_sql_query(query.query, current_user.unique_id)

        # Add user_id filter if not present
        if "WHERE" not in sql_query.upper():
            sql_query = f"{sql_query} WHERE user_id = {current_user.unique_id}"
        elif "user_id" not in sql_query:
            sql_query = sql_query.replace("WHERE", f"WHERE user_id = {current_user.unique_id} AND")

        # Execute the query asynchronously
        result = db.execute(text(sql_query))
        results = result.fetchall()

        # Convert result to list of dictionaries
        results_dict = [
    {column: value for column, value in zip(result.keys(), row)} for row in results
]

        # Synthesize a natural language explanation
        explanation = await synthesize_response(query.query, results_dict)

        return {
            "query": query.query,
            "sql_query": sql_query,
            "results": results_dict,
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )

async def generate_sql_query(query: str, user_id: str) -> str:
    """
    Calls the Groq API to generate an SQL query using the chat completions endpoint.
    """
    groq_url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    system_prompt = f"""You are an expert in generating PostgreSQL queries. The database has a table 'invoices' with columns:
    - id (integer)
    - user_id (text)
    - total_amount (float)
    - seller_name (text)
    - date_of_invoice (date)
    Always include a WHERE clause to filter by user_id = '{user_id}'. 
    Return only the SQL query without any explanation, markdown formatting, or additional text."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    payload = {
        "messages": messages,
        "model": "deepseek-r1-distill-llama-70b"
    }
    
    response = requests.post(groq_url, json=payload, headers=headers)
    
    if response.status_code == 200:
        response_data = response.json()
        content = response_data['choices'][0]['message']['content'].strip()
        
        # Extract SQL query from potential <think> blocks
        if '</think>' in content:
            content = content.split('</think>')[-1].strip()
        
        # Remove any markdown formatting
        sql_query = content.replace('sql', '').replace('', '').strip()
        
        # Ensure query ends properly (remove trailing semicolon if needed)
        if sql_query.endswith(';'):
            sql_query = sql_query[:-1]
        
        return sql_query
    else:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Groq API Error: {response.text}"
        )

async def synthesize_response(user_question: str, results: list) -> str:
    """
    Calls Groq API to generate a natural language explanation.
    """
    groq_url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    system_prompt = f"""You are a helpful assistant that explains database query results in natural language. 
    The user asked: '{user_question}'. The results are: {results}. 
    Provide a concise and clear summary in 1-2 sentences."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Please explain these results."}
    ]
    payload = {
        "messages": messages,
        "model": "deepseek-r1-distill-llama-70b"
    }
    
    response = requests.post(groq_url, json=payload, headers=headers)
    
    if response.status_code == 200:
        response_data = response.json()
        explanation = response_data['choices'][0]['message']['content'].strip()
        return explanation
    else:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Groq API Error: {response.text}"
        )