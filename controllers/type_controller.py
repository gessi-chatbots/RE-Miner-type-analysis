from fastapi import APIRouter, Query
from models.models import TypeRequest, TrainModelRequest
from services.type_service_factory import TypeServiceFactory
from services.type_service import TypeService
from enums.service_enum import TypeServiceType
import logging

router = APIRouter()

@router.get("/ping")
async def ping():
    return {"status": "ok"}


@router.post("/analyze-type")
async def analyze_type(
    request: TypeRequest,
    type_service: TypeServiceType = Query(..., alias="type-service")
):
    service = TypeServiceFactory.get_service(type_service.value)
    analyzed_reviews = service.analyze_type(request.reviews)
    return {"reviews": analyzed_reviews} 

@router.post("/train-model")
async def train_model(
    request: TrainModelRequest,
    type_service: TypeServiceType = Query(..., alias="type-service")
):
    service = TypeServiceFactory.get_service(type_service.value)
    logging.info(f"Service type: {type(service)}")
    filtered_reviews = [review for review in request.reviews if review.type != 'N/A']
    service.train_model(filtered_reviews)
    return {"status": "ok"}
