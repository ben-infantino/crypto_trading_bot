from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.src.api.routes import model_routes

app = FastAPI(
    title="Cryptocurrency Trading Bot API",
    description="API for cryptocurrency trading predictions and model management",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(model_routes.router)

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Cryptocurrency Trading Bot API",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.src.api.main:app", host="0.0.0.0", port=8000, reload=True) 