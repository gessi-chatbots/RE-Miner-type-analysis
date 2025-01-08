from enums.service_enum import TypeServiceType
from services.distilbert_type_service import DistilbertTypeService
from services.type_service_base import TypeService

class TypeServiceFactory:
    @staticmethod
    def get_service(service_type: str) -> TypeService:
        try:
            service_enum = TypeServiceType(service_type)
        except ValueError:
            available_services = ", ".join(sorted(enum.value for enum in ServiceEnum))
            raise ValueError(
                f"Invalid type service type: '{service_type}'. "
                f"Available services are: {available_services}"
            )
        
        if service_enum == TypeServiceType.DISTILBERT:
            return DistilbertTypeService()
            
        logging.error(f"Invalid service type: {service_type}")
        raise ValueError(f"Invalid service type: {service_type}")
