from fastapi import FastAPI
from app.api import recommend, ingest, analytics
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Product Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://frontend-asg.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from fastapi.responses import JSONResponse
import traceback

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    print("---- ERROR TRACEBACK ----")
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )


app.include_router(ingest.router, prefix="/api/ingest", tags=["ingest"])
app.include_router(recommend.router, prefix="/api/recommend", tags=["recommend"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])

