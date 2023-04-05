import torch
from torch import nn
from typing import List, Optional, Tuple, Union
from transformers import (PreTrainedModel,
                          AutoModel,
                          SequenceClassifierOutput,
                          TokenClassifierOutput)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


class ModelForSequenceAndTokenClassification(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_token_labels = config.num_token_labels
        self.num_sequence_labels = config.num_sequence_labels
        self.config = config

        self.bert = AutoModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)

        # For token classification
        self.token_classifier = nn.Linear(
            self.hidden_size, self.num_token_labels)

        # For the entire sequence classification
        self.sequence_classifier = nn.Linear(
            self.hidden_size, self.num_sequence_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        token_labels: Optional[torch.Tensor] = None,
        sequence_labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Union[Tuple[torch.Tensor], SequenceClassifierOutput],
            Union[Tuple[torch.Tensor], TokenClassifierOutput]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # For token classification
        token_output = outputs[0]

        token_output = self.dropout(token_output)
        token_logits = self.token_classifier(token_output)

        # For the entire sequence classification
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        sequence_logits = self.sequence_classifier(pooled_output)

        # Computing the loss as the average of both losses
        loss = None
        if token_labels is not None:
            loss_fct = CrossEntropyLoss()
            loss_tokens = loss_fct(
                token_logits.view(-1, self.num_labels), token_labels.view(-1))

            if self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss_sequence = loss_fct(
                    sequence_logits.view(-1, self.num_labels), sequence_labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss_sequence = loss_fct(sequence_logits, sequence_labels)

            loss = loss_tokens + loss_sequence

        if not return_dict:
            output = (sequence_logits, token_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=sequence_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ), TokenClassifierOutput(
            loss=loss,
            logits=token_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
