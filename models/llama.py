import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel, AutoConfig

NUM_LABELS = 4

class LlamaClassificationModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config, freeze_weights=True):
        super().__init__(config)
        self.base_model = AutoModel.from_pretrained(config._name_or_path, config=config)
        self.config = config
        self.freeze_weights = freeze_weights

        if freeze_weights:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.classifier = nn.Linear(config.hidden_size, NUM_LABELS)

    def forward(self, input_ids, attention_mask=None, labels=None):
        if self.freeze_weights:
            with torch.no_grad():
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)

        # Sum hidden states over sequence dimension
        summed_representation = outputs.last_hidden_state.sum(dim=1)
        logits = self.classifier(summed_representation)

        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels.float())

        return {"loss": loss, "logits": logits, "embeddings": summed_representation}
