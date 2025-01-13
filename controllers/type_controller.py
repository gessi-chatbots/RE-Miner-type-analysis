from fastapi import APIRouter, Query
from models.models import TypeRequest, TrainModelRequest, SingleReviewRequest, ReviewItem
from services.type_service_factory import TypeServiceFactory
from services.type_service import TypeService
from enums.service_enum import TypeServiceType
import logging
from typing import Union
from pydantic import BaseModel

router = APIRouter()

@router.get("/ping")
async def ping():
    return {"status": "ok"}


@router.post("/analyze-type")
async def analyze_type(
    request: Union[TypeRequest, SingleReviewRequest],
    type_service: TypeServiceType = Query(..., alias="type-service")
):
    service = TypeServiceFactory.get_service(type_service.value)
    
    if isinstance(request, SingleReviewRequest):
        reviews = [ReviewItem(reviewId="single", text=request.text)]
        analyzed_reviews = service.analyze_type(reviews)
        return {"reviewId": "single", "text": analyzed_reviews[0].text, "type": analyzed_reviews[0].type} 
    else:
        reviews = request.reviews
        analyzed_reviews = service.analyze_type(reviews)
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
