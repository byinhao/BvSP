from functools import partial

import torch.nn as nn
import torch
#from transformers.modeling_t5 import *
from torch.nn import BCEWithLogitsLoss
from transformers.models.t5.modeling_t5 import *
from transformers.file_utils import ModelOutput
from transformers.generation_utils import *
import torch.nn.functional as F
from transformers.generation_beam_search import *
import copy
from einops import rearrange


_CONFIG_FOR_DOC = "T5Config"

all_templates = [
    '[AT] [OT] [AC] [SP]', '[AT] [OT] [SP] [AC]', '[AT] [AC] [OT] [SP]',
    '[AT] [AC] [SP] [OT]', '[AT] [SP] [OT] [AC]', '[AT] [SP] [AC] [OT]',
    '[OT] [AT] [AC] [SP]', '[OT] [AT] [SP] [AC]', '[OT] [AC] [AT] [SP]',
    '[OT] [AC] [SP] [AT]', '[OT] [SP] [AT] [AC]', '[OT] [SP] [AC] [AT]',
    '[AC] [AT] [OT] [SP]', '[AC] [AT] [SP] [OT]', '[AC] [OT] [AT] [SP]',
    '[AC] [OT] [SP] [AT]', '[AC] [SP] [AT] [OT]', '[AC] [SP] [OT] [AT]',
    '[SP] [AT] [OT] [AC]', '[SP] [AT] [AC] [OT]', '[SP] [OT] [AT] [AC]',
    '[SP] [OT] [AC] [AT]', '[SP] [AC] [AT] [OT]', '[SP] [AC] [OT] [AT]',
    'is because is', '( , , , )'
]

PARALLELIZE_DOCSTRING = r"""
    This is an experimental feature and is a subject to change at a moment's notice.
    Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
    it will evenly distribute blocks across all devices.
    Args:
        device_map (`Dict[int, list]`, optional, defaults to None):
            A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
            automatically mapped to the first device (for esoteric reasons). That means that the first device should
            have fewer attention modules mapped to it than other devices. For reference, the t5 models have the
            following number of attention modules:
                - t5-small: 6
                - t5-base: 12
                - t5-large: 24
                - t5-3b: 24
                - t5-11b: 24
    Example:
    ```python
    # Here is an example of a device map on a machine with 4 GPUs using t5-3b, which has a total of 24 attention modules:
    model = T5ForConditionalGeneration.from_pretrained('t5-3b')
    device_map = {0: [0, 1, 2],
             1: [3, 4, 5, 6, 7, 8, 9],
             2: [10, 11, 12, 13, 14, 15, 16],
             3: [17, 18, 19, 20, 21, 22, 23]}
    model.parallelize(device_map)
    ```
"""
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.
    Example:
    ```python
    # On a 4 GPU machine with t5-3b:
    model = T5ForConditionalGeneration.from_pretrained('t5-3b')
    device_map = {0: [0, 1, 2],
                 1: [3, 4, 5, 6, 7, 8, 9],
                 2: [10, 11, 12, 13, 14, 15, 16],
                 3: [17, 18, 19, 20, 21, 22, 23]}
    model.parallelize(device_map) # Splits the model across several devices
    model.deparallelize() # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
    ```
"""

add_start_docstrings("""T5 Model with a `language modeling` head on top. """, T5_START_DOCSTRING)
@add_start_docstrings("""T5 Model with a `language modeling` head on top. """, T5_START_DOCSTRING)
class MyT5ForConditionalGeneration(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config, prefix_lenth, prefix_dropout):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.plm_modified = False
        self.prefix_lenth = prefix_lenth

        self.n_layer = self.config.num_layers
        self.n_embd = self.config.d_model
        self.n_head = self.config.num_heads
        self.n_decoder_layer = self.config.num_decoder_layers
        self.match_n_decoder_layer = self.n_decoder_layer
        self.match_n_layer = self.n_layer
        self.mid_dim = 512
        self.match_n_head = self.n_head
        self.match_n_embd = self.n_embd // self.n_head
        self.using_encoder_past_key_values, self.using_decoder_past_key_values = True, True

        self.generate_parameters()
        self.modify_plm()

        self.dropout = nn.Dropout(prefix_dropout)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.control_gradient()

    def control_gradient(self):
        # for par in self.shared.parameters():
        #     par.requires_grad = True
        # for par in self.encoder.parameters():
        #     par.requires_grad = True
        # for par in self.decoder.parameters():
        #     par.requires_grad = True
        # for par in self.lm_head.parameters():
        #     par.requires_grad = True
        
        for par in self.shared.parameters():
            par.requires_grad = False
        for par in self.encoder.parameters():
            par.requires_grad = False
        for par in self.decoder.parameters():
            par.requires_grad = False
        for par in self.lm_head.parameters():
            par.requires_grad = False

        for par in self.wte.parameters():
            par.requires_grad = True
        for par in self.control_trans.parameters():
            par.requires_grad = True

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def expand_to_batchsize_for_layer(self, tup,  batch_size, layer_id):
        return tup[layer_id].expand(-1, batch_size,-1,-1,-1)

    def get_past_key_values(self, template_types, batch_size=1):
        pvs = []
        if self.config.is_encoder_decoder and self.using_encoder_past_key_values:
            # input_tokens = self.input_tokens.unsqueeze(0).expand(batch_size, -1)
            fianl_past_key_values = []
            for ids in template_types:
                input_tokens = self.input_tokens.unsqueeze(0)
                temp_control = self.wte[ids](input_tokens)
                past_key_values = self.control_trans[ids](temp_control)  # bsz, seqlen, layer*emb
                _, seqlen, _ = past_key_values.shape
                past_key_values = past_key_values.view(1, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                       self.match_n_embd)
                past_key_values = self.dropout(past_key_values)
                past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
                if fianl_past_key_values == []:
                    for ppp in past_key_values:
                        fianl_past_key_values.append(ppp)
                else:
                    i = 0
                    while i < len(past_key_values):
                        fianl_past_key_values[i] = torch.cat((fianl_past_key_values[i], past_key_values[i]), dim=1)
                        i = i + 1
            pvs.append(fianl_past_key_values)

        else:
            pvs.append(None)

        if (not self.config.is_encoder_decoder) or self.using_decoder_past_key_values:
            fianl_past_key_values = []
            for ids in template_types:
                decoder_input_tokens = self.input_tokens.unsqueeze(0)
                decoder_temp_control = self.decoder_wte[ids](decoder_input_tokens)
                decoder_past_key_values = self.decoder_control_trans[ids](
                    decoder_temp_control)  # bsz, seqlen, layer*emb
                _, decoder_seqlen, _ = decoder_past_key_values.shape
                decoder_past_key_values = decoder_past_key_values.view(1, seqlen, self.match_n_layer * 2,
                                                                       self.match_n_head,
                                                                       self.match_n_embd)
                decoder_past_key_values = self.dropout(decoder_past_key_values)
                decoder_past_key_values = decoder_past_key_values.permute([2, 0, 3, 1, 4]).split(2)
                if fianl_past_key_values == []:
                    for ppp in decoder_past_key_values:
                        fianl_past_key_values.append(ppp)
                else:
                    i = 0
                    while i < len(decoder_past_key_values):
                        fianl_past_key_values[i] = torch.cat((fianl_past_key_values[i], decoder_past_key_values[i]),
                                                             dim=1)
                        i = i + 1
            pvs.append(fianl_past_key_values)

        else:
            pvs.append(None)
        return pvs

    def generate_parameters(self) -> None:
        r"""
        Generate parameters needed for new tokens' embedding in P-tuning
        """

        self.input_tokens = nn.Parameter(torch.arange(self.prefix_lenth).long(),
                                         requires_grad=False)  # to allow automatic devicing
        if self.config.is_encoder_decoder and self.using_encoder_past_key_values:

            self.wte = nn.ModuleList()
            self.control_trans = nn.ModuleList()

            for i in range(len(all_templates)):
                self.wte.append(nn.Embedding(self.prefix_lenth, self.n_embd))
                self.control_trans.append(nn.Sequential(
                    nn.Linear(self.n_embd, self.mid_dim),
                    nn.Tanh(),
                    # nn.Linear(self.mid_dim, self.mid_dim),
                    # nn.Tanh(),
                    nn.Linear(self.mid_dim, self.n_layer * 2 * self.n_embd)))

        if (not self.config.is_encoder_decoder) or self.using_decoder_past_key_values:

            self.decoder_wte = nn.ModuleList()
            self.decoder_control_trans = nn.ModuleList()

            for i in range(len(all_templates)):
                self.decoder_wte.append(nn.Embedding(self.prefix_lenth, self.n_embd))
                self.decoder_control_trans.append(nn.Sequential(
                    nn.Linear(self.n_embd, self.mid_dim),
                    nn.Tanh(),
                    # nn.Linear(self.mid_dim, self.mid_dim),
                    # nn.Tanh(),
                    nn.Linear(self.mid_dim, self.n_decoder_layer * 2 * self.n_embd)))

    def modify_plm(self):
        if self.plm_modified:
            return None
        if self.using_encoder_past_key_values:
            backup_encoder_forward_functions = []
            for i, layer_module in enumerate(self.encoder.block):
                backup_encoder_forward_functions.append(layer_module.layer[0].forward)
                def modified_encoder_forward(*args, **kwargs):
                    layer_id = kwargs.pop('layer_id')
                    batch_size = args[0].shape[0]
                    device = args[0].device
                    if kwargs['past_key_value'] is None:
                        kwargs['past_key_value'] = self.expand_to_batchsize_for_layer(self.our_past_key_values[0], batch_size, layer_id).to(device)
                    if kwargs['attention_mask'] is not None:
                        am = kwargs['attention_mask']  # Should check the format of the attention_mask when moving to a new plm.
                        kwargs['attention_mask'] = torch.cat([-torch.zeros((*am.shape[:-1],self.prefix_lenth), dtype = am.dtype,device=am.device), am], dim=-1)
                    return backup_encoder_forward_functions[layer_id](*args, **kwargs)
                layer_module.layer[0].forward = partial(modified_encoder_forward, layer_id=i)

        if self.using_decoder_past_key_values:
            backup_decoder_self_attn_forward_functions = []
            for i, layer_module in enumerate(self.decoder.block):
                backup_decoder_self_attn_forward_functions.append(layer_module.layer[0].forward)
                def modified_decoder_self_attn_forward(*args, **kwargs):
                    batch_size = args[0].shape[0]
                    layer_id = kwargs.pop('layer_id')
                    device = args[0].device
                    if kwargs['past_key_value'] is None:
                        kwargs['past_key_value'] = self.expand_to_batchsize_for_layer(self.our_past_key_values[1], batch_size, layer_id).to(device)
                    if kwargs['past_key_value'][0].size(-2) + args[0].size(-2) == kwargs['attention_mask'].size(-1):
                        pass
                    elif kwargs['past_key_value'][0].size(-2) + args[0].size(-2) == kwargs['attention_mask'].size(-1) +self.prefix_lenth:
                        am = kwargs['attention_mask']
                        kwargs['attention_mask'] = torch.cat([torch.zeros((*am.shape[:-1],self.prefix_lenth), dtype = am.dtype,device=am.device), am], dim=-1)
                    else:
                        raise RuntimeError("Size not match: past length: {}, inputlength:{},\
                            attention mask length {}".format(kwargs['past_key_value'][0].size(-2),
                            args[0].size(-2),kwargs['attention_mask'].size(-1)))

                    return backup_decoder_self_attn_forward_functions[layer_id](*args, **kwargs)
                layer_module.layer[0].forward = partial(modified_decoder_self_attn_forward, layer_id=i)

        self.plm_modified = True

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        template_types=None,
        **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')

            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits

            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        if "lm_labels" in kwargs:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("lm_labels")
        if "decoder_past_key_value_states" in kwargs:
            warnings.warn(
                "The `decoder_past_key_value_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_value_states")
        if "decoder_past_key_values" in kwargs:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_values")

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:

            self.our_past_key_values = self.get_past_key_values(template_types, attention_mask.shape[0])

            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output
        self.our_past_key_values = None
        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    def _prepare_encoder_decoder_kwargs_for_generation(
            self, input_ids: torch.LongTensor, model_kwargs
    ) -> Dict[str, Any]:
        if "encoder_outputs" not in model_kwargs:
            # retrieve encoder hidden states
            encoder = self.get_encoder()

            self.our_past_key_values = self.get_past_key_values(input_ids.shape[0])

            encoder_kwargs = {
                argument: value
                for argument, value in model_kwargs.items()
                if not (argument.startswith("decoder_") or argument.startswith("cross_attn"))
            }
            model_kwargs["encoder_outputs"]: ModelOutput = encoder(input_ids, return_dict=True, **encoder_kwargs)
        return model_kwargs
