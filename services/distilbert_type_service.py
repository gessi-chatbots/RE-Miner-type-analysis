from services.transformer_type_service_base import TransformerTypeServiceBase

class DistilbertTypeService(TransformerTypeServiceBase):
    def __init__(self):
        super().__init__('distilbert-base-uncased')