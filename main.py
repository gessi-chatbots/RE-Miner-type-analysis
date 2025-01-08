from fastapi import FastAPI
import logging
import sys
from controllers.type_controller import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Log to console
        # logging.FileHandler('app.log')   # Uncomment to also log to file
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI()

app.include_router(router)

if __name__ == "__main__":
    logger.info("Starting server...")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3011, reload=True, log_level="info") 