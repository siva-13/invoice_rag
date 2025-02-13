from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database.database import get_db
from services.auth import get_current_user
from services.openai_services import generate_sql_query, synthesize_response
from pydantic import BaseModel
from database.models import User
from config import DEVICE, process_pool, PDF_IMAGE_DIR, UPLOAD_DIR,MAX_WORKERS,BATCH_SIZE,RATE_LIMIT_REQUESTS,RATE_LIMIT_WINDOW,MAX_CONCURRENT_REQUESTS,API_SEMAPHORE,client,api_key



router = APIRouter(tags=["Query"])

class QueryRequest(BaseModel):
    query: str

@router.post("/query-invoices")
async def query_invoices(
    query: QueryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Process natural language queries about invoices and return results
    Example queries:
    - "What's my total invoice amount for last month?"
    - "Show me all invoices from seller X"
    - "What's the average invoice amount?"
    """
    try:
        print(query)
        # Generate SQL query from natural language
        sql_query = await generate_sql_query(query, client, current_user.unique_id)
        
        # Add user_id filter if not present (security measure)
        if "WHERE" not in sql_query.upper():
            sql_query = f"{sql_query} WHERE user_id = {current_user.unique_id}"
        elif "user_id" not in sql_query:
            sql_query = sql_query.replace("WHERE", f"WHERE user_id = {current_user.unique_id} AND")
            
        # Execute query with parameters
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: db.execute(
                text(sql_query),
                {"user_id": current_user.unique_id}
            ).fetchall()
        )
        print(f"sql response: {result}")
        # Convert result to list of dictionaries
        results = [
            {column: value for column, value in zip(row._mapping.keys(), row._mapping.values())}
            for row in result
        ]
        
        # Synthesize natural language response
        explanation = await synthesize_response(query, results, client)
        
        return {
            "query": query,
            "sql_query": sql_query,
            "results": results,
            "explanation": explanation
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )