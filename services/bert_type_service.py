from services.transformer_type_service_base import TransformerTypeServiceBase

class BertTypeService(TransformerTypeServiceBase):
    def __init__(self):
        super().__init__('bert-base-uncased') 