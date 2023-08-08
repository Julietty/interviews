import torch
import torchvision

from pytorch_metric_learning.utils.inference import InferenceModel

class EfficientNet():
    def __init__(
        self,
        embedding_size: int = 256,
        device: torch.device = torch.device('cpu'),
        data_parallel: bool = True
    ):
        # Trunk + embedder structure for pytorch_metric_learning library
        self.trunk = torchvision.models.efficientnet_b3(weights=torchvision.models.EfficientNet_B3_Weights.DEFAULT)
        trunk_output_size = self.trunk.classifier[1].in_features
        self.trunk.classifier = torch.nn.Identity()
        self.trunk.to(device)
        
        
        simple_embedder = torch.nn.Sequential(torch.nn.Linear(trunk_output_size, embedding_size))
        self.embedder = simple_embedder.to(device)

        if data_parallel:
            self.trunk = torch.nn.DataParallel(self.trunk)
            self.embedder = torch.nn.DataParallel(self.embedder)

        self.inference_model = None


    def load(
        self, 
        trunk_path: str, 
        embedder_path: str, 
        device: torch.device = torch.device('cpu')
    ) -> None:
        '''
            Load trunk and embedder weights from files.
        '''
        
        trunk_weights = torch.load(trunk_path, map_location=device)
        self.trunk.load_state_dict(trunk_weights)
        
        embedder_weights = torch.load(embedder_path, map_location=device)
        self.embedder.load_state_dict(embedder_weights)

    def set_inference_instance(self, normalize_embeddings: bool = True):
        self.inference_model = InferenceModel(
            trunk=self.trunk,
            embedder=self.embedder,
            normalize_embeddings=normalize_embeddings,
        )



