import torch
import torch.nn as nn
import timm
from transformers import AutoModel, AutoConfig


class TimmFeatureExtractor(nn.Module):
    """
    Wrapper for models available in the pytorch-image-models (timm) library.
    Target models: SigLIP, DINOv2, EVA-CLIP.
    """
    def __init__(self, model_name: str, pretrained: bool = True, dynamic_img_size: bool = False, **kwargs):
        super().__init__()
        # num_classes=0 removes the classification head and applies Global Average Pooling (GAP)
        # or selects the CLS token automatically depending on the model architecture.
        if "bvra" in model_name.lower():
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
            )
            self.model.head = torch.nn.Identity()
        else:
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                dynamic_img_size=dynamic_img_size,
            )
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Channels, Height, Width)
        Returns:
            Tensor of shape (Batch, Embedding_Dimension)
        """
        return self.model(x)


class HuggingFaceFeatureExtractor(nn.Module):
    """
    Wrapper for models hosted on HuggingFace Hub that require custom handling.
    Target models: AIMv2, AM-RADIO.
    """
    def __init__(
            self,
            model_repo: str,
            trust_remote_code: bool = True,
            dynamic_img_size: bool = False,
            revision: str = None,
            use_f16: bool = False,
            use_low_cpu: bool = False,
            **kwargs
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_repo, trust_remote_code=trust_remote_code)
        self.dynamic_img_size = dynamic_img_size
        if "apple/aimv2" in model_repo:
            self.model = AutoModel.from_pretrained(
                model_repo,
                trust_remote_code=trust_remote_code,
                revision=revision,
                dtype=torch.float16 if use_f16 else torch.float32,
                low_cpu_mem_usage=use_low_cpu,
            )
        else:
            self.model = AutoModel.from_pretrained(
                model_repo,
                config=self.config,
                trust_remote_code=trust_remote_code,
                dtype=torch.float16 if use_f16 else torch.float32,
                low_cpu_mem_usage=use_low_cpu,
            )
        if hasattr(self.model, "vision_model"):
            self.model = self.model.vision_model

        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Channels, Height, Width)
        Returns:
            Tensor of shape (Batch, Embedding_Dimension)
        """
        if self.dynamic_img_size:
            outputs = self.model(
                pixel_values=x,
                interpolate_pos_encoding=self.dynamic_img_size
            )
        else:
            outputs = self.model(x)

        # If the output is a tuple/list, use the first Tensor that has 2+ dims
        if isinstance(outputs, (tuple, list)):
            for o in outputs:
                if isinstance(o, torch.Tensor) and o.ndim >= 2:
                    return o
            raise ValueError("No suitable tensor found in tuple/list outputs")

        # If the output is a dataclass or object with tensor attributes
        # Common candidates: pooler_output, last_hidden_state, features, embedding, hidden_states
        candidate_attrs = [
            "pooler_output",
            "last_hidden_state",
            "features",
            "embedding",
            "hidden_states"
        ]
        for attr in candidate_attrs:
            if hasattr(outputs, attr):
                tensor = getattr(outputs, attr)
                # For hidden_states list/tuple, pick last layer
                if isinstance(tensor, (tuple, list)):
                    tensor = tensor[-1]
                if isinstance(tensor, torch.Tensor):
                    # Use CLS token if last_hidden_state
                    if attr == "last_hidden_state" and tensor.ndim == 3:
                        return tensor[:, 0]
                    return tensor

        # Fallback: if outputs itself is a tensor
        if isinstance(outputs, torch.Tensor):
            return outputs

        raise ValueError(f"Cannot extract embedding tensor from outputs of type {type(outputs)}")


class TorchHubFeatureExtractor(nn.Module):
    """
    Wrapper for models loaded via torch.hub.
    Target models: Franca, custom research models, etc.
    """
    def __init__(self, model_repo: str = None, pretrained: bool = True, **kwargs):
        """
        Args:
            repo (str): Torch Hub repo, e.g. "valeoai/Franca"
            model_name (str): Model name inside the repo (e.g., "franca_vitg14")
            pretrained (bool): Load pretrained weights if available.
            kwargs: Additional arguments passed to torch.hub.load
        """
        super().__init__()
        repo, model_name = model_repo.split(":")

        # torch.hub.load(repo, model_name, pretrained=True)
        if model_name is None:
            raise ValueError("TorchHub models require a model_name")

        self.model = torch.hub.load(
            repo,
            model_name,
            pretrained=pretrained,
        )

        # Put model in eval mode
        self.model.eval()

        # Remove classifier if present
        self.model = self._remove_head(self.model)

    def _remove_head(self, model):
        """
        Removes classification layers for common architectures:
        - vision transformers (like Franca)
        - CNNs (resnets, etc.)
        """

        # Case 1: Vision Transformers with attribute `head` or `classifier`
        if hasattr(model, "head"):
            try:
                model.head = nn.Identity()
                return model
            except Exception:
                pass

        if hasattr(model, "classifier"):
            try:
                model.classifier = nn.Identity()
                return model
            except Exception:
                pass

        # Case 2: timm-like models with fc
        if hasattr(model, "fc"):
            try:
                model.fc = nn.Identity()
                return model
            except Exception:
                pass

        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Channels, Height, Width)
        Returns:
            Feature tensor of shape (Batch, Embedding_Dimension)
        """
        outputs = self.model(x)

        # Some models return dicts or have special forward paths
        if isinstance(outputs, dict):
            # Common fallback: use "embedding" or "features"
            for key in ["embedding", "features", "feature", "pooled"]:
                if key in outputs:
                    return outputs[key]

            # Otherwise fallback to first value
            return next(iter(outputs.values()))

        # If output is a tuple → return first element
        if isinstance(outputs, (tuple, list)):
            return outputs[0]

        return outputs
