import torch
import torch.nn as nn


class EncoderModel(nn.Module):

    def __init__(self, args, base_model):
        super().__init__()
        self.args = args
        self.base_model = base_model
        self.num_classes = args.num_classes
        self.method = args.method
        self.dense = nn.Linear(base_model.config.hidden_size, base_model.config.hidden_size)
        if base_model.config.classifier_dropout is not None:
            classifier_dropout = base_model.config.classifier_dropout
        else:
            classifier_dropout = base_model.config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(base_model.config.hidden_size, args.num_classes)

        for param in base_model.parameters():
            param.requires_grad_(True)

    def forward(self, batch):
        # if self.args.model_name == 'bert':
        #     raw_outputs = self.base_model(**batch)
        # else:
        #     raw_outputs = self.base_model(input_ids=batch['input_ids'],
        #                                   attention_mask=batch['attention_mask'],
        #                                   position_ids=batch['position_ids'] if self.method == 'dualcl' else None)

        raw_outputs = self.base_model(**batch)

        hiddens = raw_outputs.last_hidden_state
        cls_feats = hiddens[:, 0, :]
        if self.method in ['ce', 'scl']:
            label_feats = None
            x = self.dropout(cls_feats)
            x = self.dense(x)
            x = torch.tanh(x)
            x = self.dropout(x)
            predicts = self.out_proj(x)
        else:
            label_feats = hiddens[:, 1:self.num_classes + 1, :]
            predicts = torch.einsum('bd,bcd->bc', cls_feats, label_feats)

        outputs = {
            'predicts': predicts,
            'cls_feats': cls_feats,
            'label_feats': label_feats
        }
        return outputs
