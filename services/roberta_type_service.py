from services.transformer_type_service_base import TransformerTypeServiceBase

class RobertaTypeService(TransformerTypeServiceBase):
    def __init__(self):
        super().__init__('roberta-base') 