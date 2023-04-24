"""
This module include code to perform SELD task.
output rates:
  - feature: has feature rates
  - gt: has output/label rate
"""
import copy
import os
import shutil

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from transformers.activations import ACT2FN
from transformers import AutoConfig, Wav2Vec2ForCTC
from typing import Optional, Tuple, Union
from models.interfaces import BaseModel
from models.stereonet_utils import PositionalConvEmbedding, Wav2Vec2FeedForward
from models.model_utils import init_layer, interpolate_tensor

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)

class DummyWav2Vec2Model(Wav2Vec2PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)

class TrueStereoNet(BaseModel):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, sed_threshold: float = 0.3, doa_threshold: int = 20,
                 label_rate: int = 10, feature_rate: int = None, optimizer_name: str = 'Adam', lr: float = 1e-3,
                 loss_weight: Tuple = None, output_pred_dir: str = None, submission_dir: str = None,
                 test_chunk_len: int = None, test_chunk_hop_len: int = None, gt_meta_root_dir: str = None,
                 output_format: str = None, eval_version: str = '2021', is_eval: bool = False, stereonet_cfg_path=None, **kwargs):
        super().__init__(sed_threshold=sed_threshold, doa_threshold=doa_threshold, label_rate=label_rate,
                         feature_rate=feature_rate, optimizer_name=optimizer_name, lr=lr,
                         output_pred_dir=output_pred_dir, submission_dir=submission_dir, test_chunk_len=test_chunk_len,
                         test_chunk_hop_len=test_chunk_hop_len, gt_meta_root_dir=gt_meta_root_dir,
                         output_format=output_format, eval_version=eval_version)
        self.save_hyperparameters()
        self.encoder_sed = encoder
        self.encoder_doa = copy.deepcopy(encoder)
        # self.decoder = decoder
        self.loss_weight = loss_weight
        self.time_downsample_ratio = float(self.encoder_sed.time_downsample_ratio)
        self.n_classes = decoder.n_classes
        self.doa_format = decoder.doa_format  # doa_format redundance since we have doa_output_format
        self.is_eval = is_eval

        self.seld_val = None

        self.config = AutoConfig.from_pretrained(stereonet_cfg_path)


        self.encoder_proj1 = StereoNetFeatureProjection(self.config)
        self.encoder_proj2 = StereoNetFeatureProjection(self.config)
        self.stereonet = StereoNetDecoder(self.config, self.n_classes)
        # self.dummy1 = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        # self.dummy2 = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        # self.stereonet.update_weights(self.dummy1, self.dummy2)
        # del(self.dummy1)
        # del(self.dummy2)



    def forward(self, x):
        """
        x: (batch_size, n_channels, n_timesteps (n_frames), n_features).
        """
        # print(f'input: {x.shape}')
        x_sed = self.encoder_sed(x)  # (batch_size, n_channels, n_timesteps, n_features)
        x_sed = torch.mean(x_sed, dim=3)
        x_sed = x_sed.transpose(1, 2)
        # print(f'after encoder: {x.shape}')
        hidden_states1 = self.encoder_proj1(x_sed)[0]
        # print(f'hidden_states1 = {hidden_states1.shape}')

        x_doa = self.encoder_doa(x)  # (batch_size, n_channels, n_timesteps, n_features)
        x_doa = torch.mean(x_doa, dim=3)
        x_doa = x_doa.transpose(1, 2)
        hidden_states2 = self.encoder_proj2(x_doa)[0]

        hidden_states = torch.cat((hidden_states1.unsqueeze(dim=1), hidden_states2.unsqueeze(dim=1)), dim=1)
        # print(f'hidden_states = {hidden_states.shape}')
        output_dict = self.stereonet(hidden_states)
        # output_dict = {
        #     'event_frame_logit': event_frame_logit, # (batch_size, n_timesteps, n_classes)
        #     'doa_frame_output': doa_output, # (batch_size, n_timesteps, 3* n_classes)
        # }
        return output_dict

    def common_step(self, batch_data):
        x, y_sed, y_doa, _ = batch_data
        # target dict has frame_rate = label_rate
        target_dict = {
            'event_frame_gt': y_sed,
            'doa_frame_gt': y_doa,
        }
        # forward
        pred_dict = self.forward(x)
        # print(f'event_frame_logit.shape={pred_dict["event_frame_logit"].shape}')
        # print(f'doa_frame_output.shape={pred_dict["doa_frame_output"].shape}')
        # interpolate output to match label rate
        pred_dict['event_frame_logit'] = interpolate_tensor(
            pred_dict['event_frame_logit'], ratio=self.time_downsample_ratio * self.label_rate / self.feature_rate)
        pred_dict['doa_frame_output'] = interpolate_tensor(
            pred_dict['doa_frame_output'], ratio=self.time_downsample_ratio * self.label_rate / self.feature_rate)
        # print(f'interpolated event_frame_logit.shape={pred_dict["event_frame_logit"].shape}')
        # print(f'interpolated doa_frame_output.shape={pred_dict["doa_frame_output"].shape}')
        return target_dict, pred_dict

    def training_step(self, train_batch, batch_idx):
        target_dict, pred_dict = self.common_step(train_batch)
        loss, sed_loss, doa_loss = self.compute_loss(target_dict=target_dict, pred_dict=pred_dict)
        # logging
        self.log('trl', loss, prog_bar=True, logger=True)
        self.log('trsl', sed_loss, prog_bar=True, logger=True)
        self.log('trdl', doa_loss, prog_bar=True, logger=True)
        training_step_outputs = {'loss': loss}
        return training_step_outputs

    def training_epoch_end(self, training_step_outputs):
        # clear temp folder to write val output
        if self.submission_dir is not None:
            shutil.rmtree(self.submission_dir, ignore_errors=True)
            os.makedirs(self.submission_dir, exist_ok=True)

    def validation_step(self, val_batch, batch_idx):
        target_dict, pred_dict = self.common_step(val_batch)
        loss, sed_loss, doa_loss = self.compute_loss(target_dict=target_dict, pred_dict=pred_dict)
        # write output file
        filenames = val_batch[-1]
        self.write_output_submission(pred_dict=pred_dict, filenames=filenames)
        # logging
        self.log('vall', loss, prog_bar=True, logger=True)
        self.log('valsl', sed_loss, prog_bar=True, logger=True)
        self.log('valdl', doa_loss, prog_bar=True, logger=True)

    def validation_epoch_end(self, validation_step_outputs):
        # Get list of csv filename
        pred_filenames = os.listdir(self.submission_dir)
        pred_filenames = [fn for fn in pred_filenames if fn.endswith('csv')]
        # Compute validation metrics
        ER, F1, LE, LR, seld_error = self.evaluate_output_prediction_csv(pred_filenames=pred_filenames)
        # log metrics
        self.log('valER', ER)
        self.log('valF1', F1)
        self.log('valLE', LE)
        self.log('valLR', LR)
        self.log('valSeld', seld_error)
        self.lit_logger.info('Epoch {} - Validation - SELD: {:.4f} - SED ER: {:.4f} - F1: {:.4f} - DOA LE: {:.4f} - '
                             'LR: {:.4f}'.format(self.current_epoch, seld_error, ER, F1, LE, LR))

    def test_step(self, test_batch, batch_idx):
        target_dict, pred_dict = self.common_test_step(test_batch)
        # write output submission
        filenames = test_batch[-1]
        self.write_output_submission(pred_dict=pred_dict, filenames=filenames)
        # write output prediction
        if self.output_pred_dir:
            self.write_output_prediction(pred_dict=pred_dict, target_dict=target_dict, filenames=filenames)

    def test_epoch_end(self, test_step_outputs):
        pred_filenames = os.listdir(self.submission_dir)
        pred_filenames = [fn for fn in pred_filenames if fn.endswith('csv')]
        self.lit_logger.info('Number of test files: {}'.format(len(pred_filenames)))
        # Compute validation metrics
        if self.is_eval:
            ER, F1, LE, LR, seld_error = 0.0, 0.0, 0.0, 0.0, 0.0
        else:
            ER, F1, LE, LR, seld_error = self.evaluate_output_prediction_csv(pred_filenames=pred_filenames)
        # log metrics
        self.log('valER', ER)
        self.log('valF1', F1)
        self.log('valLE', LE)
        self.log('valLR', LR)
        self.log('valSeld', seld_error)
        self.lit_logger.info('Epoch {} - Test - SELD: {:.4f} - SED ER: {:.4f} - F1: {:.4f} - DOA LE: {:.4f} - '
                             'LR: {:.4f}'.format(self.current_epoch, seld_error, ER, F1, LE, LR))

    def common_test_step(self, batch_data):
        x, y_sed, y_doa, _ = batch_data
        # target dict has frame_rate = label_rate
        target_dict = {
            'event_frame_gt': y_sed,
            'doa_frame_gt': y_doa,
        }
        # forward
        pred_dict = self.forward(x)
        # interpolate output to match label rate
        pred_dict['event_frame_logit'] = interpolate_tensor(
            pred_dict['event_frame_logit'], ratio=self.time_downsample_ratio * self.label_rate / self.feature_rate)
        pred_dict['doa_frame_output'] = interpolate_tensor(
            pred_dict['doa_frame_output'], ratio=self.time_downsample_ratio * self.label_rate / self.feature_rate)

        return target_dict, pred_dict

class StereoNet(BaseModel):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, sed_threshold: float = 0.3, doa_threshold: int = 20,
                 label_rate: int = 10, feature_rate: int = None, optimizer_name: str = 'Adam', lr: float = 1e-3,
                 loss_weight: Tuple = None, output_pred_dir: str = None, submission_dir: str = None,
                 test_chunk_len: int = None, test_chunk_hop_len: int = None, gt_meta_root_dir: str = None,
                 output_format: str = None, eval_version: str = '2021', is_eval: bool = False, stereonet_cfg_path=None, **kwargs):
        super().__init__(sed_threshold=sed_threshold, doa_threshold=doa_threshold, label_rate=label_rate,
                         feature_rate=feature_rate, optimizer_name=optimizer_name, lr=lr,
                         output_pred_dir=output_pred_dir, submission_dir=submission_dir, test_chunk_len=test_chunk_len,
                         test_chunk_hop_len=test_chunk_hop_len, gt_meta_root_dir=gt_meta_root_dir,
                         output_format=output_format, eval_version=eval_version)
        self.save_hyperparameters()
        self.encoder = encoder
        # self.decoder = decoder
        self.loss_weight = loss_weight
        self.time_downsample_ratio = float(self.encoder.time_downsample_ratio)
        self.n_classes = decoder.n_classes
        self.doa_format = decoder.doa_format  # doa_format redundance since we have doa_output_format
        self.is_eval = is_eval

        self.seld_val = None

        self.config = AutoConfig.from_pretrained(stereonet_cfg_path)


        self.encoder_proj1 = StereoNetFeatureProjection(self.config)
        self.encoder_proj2 = StereoNetFeatureProjection(self.config)
        self.stereonet = StereoNetDecoder(self.config, self.n_classes)
        self.dummy1 = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        self.dummy2 = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        self.stereonet.update_weights(self.dummy1, self.dummy2)
        del(self.dummy1)
        del(self.dummy2)



    def forward(self, x):
        """
        x: (batch_size, n_channels, n_timesteps (n_frames), n_features).
        """
        # print(f'input: {x.shape}')
        x = self.encoder(x)  # (batch_size, n_channels, n_timesteps, n_features)
        x = torch.mean(x, dim=3)
        x = x.transpose(1, 2)
        # print(f'after encoder: {x.shape}')
        hidden_states1 = self.encoder_proj1(x)[0]
        # print(f'hidden_states1 = {hidden_states1.shape}')
        hidden_states2 = self.encoder_proj2(x)[0]
        hidden_states = torch.cat((hidden_states1.unsqueeze(dim=1), hidden_states2.unsqueeze(dim=1)), dim=1)
        # print(f'hidden_states = {hidden_states.shape}')
        output_dict = self.stereonet(hidden_states)
        # output_dict = {
        #     'event_frame_logit': event_frame_logit, # (batch_size, n_timesteps, n_classes)
        #     'doa_frame_output': doa_output, # (batch_size, n_timesteps, 3* n_classes)
        # }
        return output_dict

    def common_step(self, batch_data):
        x, y_sed, y_doa, _ = batch_data
        # target dict has frame_rate = label_rate
        target_dict = {
            'event_frame_gt': y_sed,
            'doa_frame_gt': y_doa,
        }
        # forward
        pred_dict = self.forward(x)
        # print(f'event_frame_logit.shape={pred_dict["event_frame_logit"].shape}')
        # print(f'doa_frame_output.shape={pred_dict["doa_frame_output"].shape}')
        # interpolate output to match label rate
        pred_dict['event_frame_logit'] = interpolate_tensor(
            pred_dict['event_frame_logit'], ratio=self.time_downsample_ratio * self.label_rate / self.feature_rate)
        pred_dict['doa_frame_output'] = interpolate_tensor(
            pred_dict['doa_frame_output'], ratio=self.time_downsample_ratio * self.label_rate / self.feature_rate)
        # print(f'interpolated event_frame_logit.shape={pred_dict["event_frame_logit"].shape}')
        # print(f'interpolated doa_frame_output.shape={pred_dict["doa_frame_output"].shape}')
        return target_dict, pred_dict

    def training_step(self, train_batch, batch_idx):
        target_dict, pred_dict = self.common_step(train_batch)
        loss, sed_loss, doa_loss = self.compute_loss(target_dict=target_dict, pred_dict=pred_dict)
        # logging
        self.log('trl', loss, prog_bar=True, logger=True)
        self.log('trsl', sed_loss, prog_bar=True, logger=True)
        self.log('trdl', doa_loss, prog_bar=True, logger=True)
        training_step_outputs = {'loss': loss}
        return training_step_outputs

    def training_epoch_end(self, training_step_outputs):
        # clear temp folder to write val output
        if self.submission_dir is not None:
            shutil.rmtree(self.submission_dir, ignore_errors=True)
            os.makedirs(self.submission_dir, exist_ok=True)

    def validation_step(self, val_batch, batch_idx):
        target_dict, pred_dict = self.common_step(val_batch)
        loss, sed_loss, doa_loss = self.compute_loss(target_dict=target_dict, pred_dict=pred_dict)
        # write output file
        filenames = val_batch[-1]
        self.write_output_submission(pred_dict=pred_dict, filenames=filenames)
        # logging
        self.log('vall', loss, prog_bar=True, logger=True)
        self.log('valsl', sed_loss, prog_bar=True, logger=True)
        self.log('valdl', doa_loss, prog_bar=True, logger=True)

    def validation_epoch_end(self, validation_step_outputs):
        # Get list of csv filename
        pred_filenames = os.listdir(self.submission_dir)
        pred_filenames = [fn for fn in pred_filenames if fn.endswith('csv')]
        # Compute validation metrics
        ER, F1, LE, LR, seld_error = self.evaluate_output_prediction_csv(pred_filenames=pred_filenames)
        # log metrics
        self.log('valER', ER)
        self.log('valF1', F1)
        self.log('valLE', LE)
        self.log('valLR', LR)
        self.log('valSeld', seld_error)
        self.lit_logger.info('Epoch {} - Validation - SELD: {:.4f} - SED ER: {:.4f} - F1: {:.4f} - DOA LE: {:.4f} - '
                             'LR: {:.4f}'.format(self.current_epoch, seld_error, ER, F1, LE, LR))

    def test_step(self, test_batch, batch_idx):
        target_dict, pred_dict = self.common_test_step(test_batch)
        # write output submission
        filenames = test_batch[-1]
        self.write_output_submission(pred_dict=pred_dict, filenames=filenames)
        # write output prediction
        if self.output_pred_dir:
            self.write_output_prediction(pred_dict=pred_dict, target_dict=target_dict, filenames=filenames)

    def test_epoch_end(self, test_step_outputs):
        pred_filenames = os.listdir(self.submission_dir)
        pred_filenames = [fn for fn in pred_filenames if fn.endswith('csv')]
        self.lit_logger.info('Number of test files: {}'.format(len(pred_filenames)))
        # Compute validation metrics
        if self.is_eval:
            ER, F1, LE, LR, seld_error = 0.0, 0.0, 0.0, 0.0, 0.0
        else:
            ER, F1, LE, LR, seld_error = self.evaluate_output_prediction_csv(pred_filenames=pred_filenames)
        # log metrics
        self.log('valER', ER)
        self.log('valF1', F1)
        self.log('valLE', LE)
        self.log('valLR', LR)
        self.log('valSeld', seld_error)
        self.lit_logger.info('Epoch {} - Test - SELD: {:.4f} - SED ER: {:.4f} - F1: {:.4f} - DOA LE: {:.4f} - '
                             'LR: {:.4f}'.format(self.current_epoch, seld_error, ER, F1, LE, LR))

    def common_test_step(self, batch_data):
        x, y_sed, y_doa, _ = batch_data
        # target dict has frame_rate = label_rate
        target_dict = {
            'event_frame_gt': y_sed,
            'doa_frame_gt': y_doa,
        }
        # forward
        pred_dict = self.forward(x)
        # interpolate output to match label rate
        pred_dict['event_frame_logit'] = interpolate_tensor(
            pred_dict['event_frame_logit'], ratio=self.time_downsample_ratio * self.label_rate / self.feature_rate)
        pred_dict['doa_frame_output'] = interpolate_tensor(
            pred_dict['doa_frame_output'], ratio=self.time_downsample_ratio * self.label_rate / self.feature_rate)

        return target_dict, pred_dict



class MultiHeadAttentionLayer(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            mode: str = 'self',
            key_value_states: Optional[torch.Tensor] = None,
            _key_states: Optional[torch.Tensor] = None,
            _value_states: Optional[torch.Tensor] = None,
            _query_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        if(mode == 'self'):
            _key_states = hidden_states
            _value_states = hidden_states
            _query_states = hidden_states
        elif(mode == 'cross'):
            _key_states = key_value_states
            _value_states = key_value_states
            _query_states = hidden_states
        elif(mode == 'selfdoc'):
            _key_states = _key_states
            _value_states = _value_states
            _query_states = _query_states

        # print('start')
        # print(f'key_states: {_key_states}')
        # print(f'value_states: {_value_states}')
        # print(f'query_states: {_query_states}')
        # print('end')

        bsz, tgt_len, _ = hidden_states.size()

        # get query, key, val proj
        query_states = self.q_proj(_query_states) * self.scaling
        key_states = self._shape(self.k_proj(_key_states), -1, bsz)
        value_states = self._shape(self.v_proj(_value_states), -1, bsz)


        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )


        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output



class SelfDocLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_layers1 = nn.ModuleList([MultiHeadAttentionLayer(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,) for _ in range(4)])
        self.attn_layers2 = nn.ModuleList([MultiHeadAttentionLayer(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,) for _ in range(4)])

        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward1 = Wav2Vec2FeedForward(config)
        self.final_layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward2 = Wav2Vec2FeedForward(config)
        self.final_layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)


    def forward(self, hidden_states: torch.Tensor,):
        hidden_states1 = hidden_states[:, 0, :, :]
        hidden_states2 = hidden_states[:, 1, :, :]

        attn_residual1 = hidden_states1
        hidden_states1 = self.layer_norm1(hidden_states1)
        attn_residual2 = hidden_states2
        hidden_states2 = self.layer_norm2(hidden_states2)
        hidden_states1 = self.attn_layers1[0](hidden_states1, mode='selfdoc', _key_states=hidden_states2, _value_states=hidden_states1, _query_states=hidden_states1)
        hidden_states2 = self.attn_layers2[0](hidden_states2, mode='selfdoc', _key_states=hidden_states1, _value_states=hidden_states2, _query_states=hidden_states2)
        hidden_states1 = self.attn_layers1[1](hidden_states1)
        hidden_states2 = self.attn_layers2[1](hidden_states2)
        hidden_states1 = self.attn_layers1[2](hidden_states1, mode='selfdoc', _key_states=hidden_states1, _value_states=hidden_states2, _query_states=hidden_states1)
        hidden_states2 = self.attn_layers2[2](hidden_states2, mode='selfdoc', _key_states=hidden_states2, _value_states=hidden_states1, _query_states=hidden_states2)
        hidden_states1 = self.attn_layers1[3](hidden_states1)
        hidden_states2 = self.attn_layers2[3](hidden_states2)


        hidden_states1 = self.dropout(hidden_states1)
        hidden_states1 = attn_residual1 + hidden_states1
        hidden_states1 = hidden_states1 + self.feed_forward1(self.final_layer_norm1(hidden_states1))


        hidden_states2 = self.dropout(hidden_states2)
        hidden_states2 = attn_residual2 + hidden_states2
        hidden_states2 = hidden_states2 + self.feed_forward2(self.final_layer_norm2(hidden_states2))

        hidden_states = torch.cat((hidden_states1.unsqueeze(dim=1), hidden_states2.unsqueeze(dim=1)), dim=1)

        outputs = (hidden_states,)

        return outputs

class EncoderSubLayer(nn.Module):
    def __init__(self, config, cross_attention=False):
        super().__init__()
        self.attention1 = MultiHeadAttentionLayer(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
        )
        self.attention2 = MultiHeadAttentionLayer(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward1 = Wav2Vec2FeedForward(config)
        self.final_layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward2 = Wav2Vec2FeedForward(config)
        self.final_layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.cross_attention = cross_attention

    def forward(
        self,
        hidden_states: torch.Tensor
    ):
        _hidden_states1 = hidden_states[:,0,:,:]
        _hidden_states2 = hidden_states[:,1,:,:]

        attn_residual1 = _hidden_states1
        _hidden_states1 = self.layer_norm1(_hidden_states1)
        attn_residual2 = _hidden_states2
        _hidden_states2 = self.layer_norm2(_hidden_states2)

        if(self.cross_attention):
            hidden_states1 = self.attention1(_hidden_states1, mode='cross', key_value_states=_hidden_states2)
        else:
            hidden_states1 = self.attention1(_hidden_states1)

        hidden_states1 = self.dropout(hidden_states1)
        hidden_states1 = attn_residual1 + hidden_states1
        hidden_states1 = hidden_states1 + self.feed_forward1(self.final_layer_norm1(hidden_states1))


        if(self.cross_attention):
            hidden_states2 = self.attention2(_hidden_states2, mode='cross', key_value_states=_hidden_states1)
        else:
            hidden_states2 = self.attention2(_hidden_states2)
        hidden_states2 = self.dropout(hidden_states2)
        hidden_states2 = attn_residual2 + hidden_states2
        hidden_states2 = hidden_states2 + self.feed_forward2(self.final_layer_norm2(hidden_states2))


        hidden_states = torch.cat((hidden_states1.unsqueeze(dim=1), hidden_states2.unsqueeze(dim=1)), dim=1)

        outputs = (hidden_states, )

        return outputs

class StereoNetFeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states):
        # non-projected hidden states are needed for quantization
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states

class StereoNetDecoder(nn.Module):
    def __init__(self, config, n_classes):
        super().__init__()
        self.config = config
        self.pos_conv_embed1 = PositionalConvEmbedding(config)
        self.pos_conv_embed2 = PositionalConvEmbedding(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = [EncoderSubLayer(config, cross_attention=False) for _ in range(config.num_hidden_layers)]
        for i in range(config.num_hidden_layers):
            if (i==0 or i==2):
                # self.layers[i] = SelfDocLayer(config)
                self.layers[i] = EncoderSubLayer(config, cross_attention=True)
        self.layers = nn.ModuleList(self.layers)
        self.gradient_checkpointing = False


        self.fc_size = config.output_hidden_size
        self.n_classes = n_classes
        # sed
        self.event_fc_1 = nn.Linear(self.fc_size, self.fc_size // 2, bias=True)
        self.event_dropout_1 = nn.Dropout(p=0.2)
        self.event_fc_2 = nn.Linear(self.fc_size // 2, self.n_classes, bias=True)
        self.event_dropout_2 = nn.Dropout(p=0.2)

        # doa
        self.x_fc_1 = nn.Linear(self.fc_size, self.fc_size // 2, bias=True)
        self.y_fc_1 = nn.Linear(self.fc_size, self.fc_size // 2, bias=True)
        self.z_fc_1 = nn.Linear(self.fc_size, self.fc_size // 2, bias=True)
        self.x_dropout_1 = nn.Dropout(p=0.2)
        self.y_dropout_1 = nn.Dropout(p=0.2)
        self.z_dropout_1 = nn.Dropout(p=0.2)
        self.x_fc_2 = nn.Linear(self.fc_size // 2, self.n_classes, bias=True)
        self.y_fc_2 = nn.Linear(self.fc_size // 2, self.n_classes, bias=True)
        self.z_fc_2 = nn.Linear(self.fc_size // 2, self.n_classes, bias=True)
        self.x_dropout_2 = nn.Dropout(p=0.2)
        self.y_dropout_2 = nn.Dropout(p=0.2)
        self.z_dropout_2 = nn.Dropout(p=0.2)

        self.init_weights()

    def init_weights(self):
        init_layer(self.event_fc_1)
        init_layer(self.event_fc_2)
        init_layer(self.x_fc_1)
        init_layer(self.y_fc_1)
        init_layer(self.z_fc_1)
        init_layer(self.x_fc_2)
        init_layer(self.y_fc_2)
        init_layer(self.z_fc_2)

    def update_weights(self, dummy1, dummy2):
        dummy2.load_state_dict(dummy1.state_dict())
        # self.pos_conv_embed1 = dummy1.wav2vec2.encoder.pos_conv_embed
        # self.pos_conv_embed2 = dummy2.wav2vec2.encoder.pos_conv_embed
        # for i, layer in enumerate(self.layers):
        #     self.layers[i].attention1.k_proj = dummy1.wav2vec2.encoder.layers[i].attention.k_proj
        #     self.layers[i].attention1.v_proj = dummy1.wav2vec2.encoder.layers[i].attention.v_proj
        #     self.layers[i].attention1.q_proj = dummy1.wav2vec2.encoder.layers[i].attention.q_proj
        #     self.layers[i].attention1.out_proj = dummy1.wav2vec2.encoder.layers[i].attention.out_proj
        #     self.layers[i].feed_forward1 = dummy1.wav2vec2.encoder.layers[i].feed_forward
        #
        #     self.layers[i].attention2.k_proj = dummy2.wav2vec2.encoder.layers[i].attention.k_proj
        #     self.layers[i].attention2.v_proj = dummy2.wav2vec2.encoder.layers[i].attention.v_proj
        #     self.layers[i].attention2.q_proj = dummy2.wav2vec2.encoder.layers[i].attention.q_proj
        #     self.layers[i].attention2.out_proj = dummy2.wav2vec2.encoder.layers[i].attention.out_proj
        #     self.layers[i].feed_forward2 = dummy2.wav2vec2.encoder.layers[i].feed_forward

        self.pos_conv_embed1.load_state_dict(dummy1.wav2vec2.encoder.pos_conv_embed.state_dict())
        self.pos_conv_embed2.load_state_dict(dummy2.wav2vec2.encoder.pos_conv_embed.state_dict())
        self.layer_norm1.load_state_dict(dummy1.wav2vec2.encoder.layer_norm.state_dict())
        self.layer_norm2.load_state_dict(dummy2.wav2vec2.encoder.layer_norm.state_dict())
        for i, layer in enumerate(self.layers):
            self.layers[i].attention1.load_state_dict(dummy1.wav2vec2.encoder.layers[i].attention.state_dict())
            self.layers[i].attention2.load_state_dict(dummy2.wav2vec2.encoder.layers[i].attention.state_dict())
            self.layers[i].layer_norm1.load_state_dict(dummy1.wav2vec2.encoder.layers[i].layer_norm.state_dict())
            self.layers[i].layer_norm2.load_state_dict(dummy2.wav2vec2.encoder.layers[i].layer_norm.state_dict())
            self.layers[i].feed_forward1.load_state_dict(dummy1.wav2vec2.encoder.layers[i].feed_forward.state_dict())
            self.layers[i].feed_forward2.load_state_dict(dummy2.wav2vec2.encoder.layers[i].feed_forward.state_dict())
            self.layers[i].final_layer_norm1.load_state_dict(dummy1.wav2vec2.encoder.layers[i].final_layer_norm.state_dict())
            self.layers[i].final_layer_norm2.load_state_dict(dummy2.wav2vec2.encoder.layers[i].final_layer_norm.state_dict())


    def forward(
        self,
        hidden_states
    ):

        hidden_states1 = hidden_states[:, 0, :, :]
        hidden_states2 = hidden_states[:, 1, :, :]
        position_embeddings1 = self.pos_conv_embed1(hidden_states1)
        hidden_states1 = hidden_states1 + position_embeddings1
        hidden_states1 = self.dropout(hidden_states1)

        position_embeddings2 = self.pos_conv_embed2(hidden_states2)
        hidden_states2 = hidden_states2 + position_embeddings2
        hidden_states2 = self.dropout(hidden_states2)
        hidden_states = torch.cat((hidden_states1.unsqueeze(dim=1), hidden_states2.unsqueeze(dim=1)), dim=1)

        for i, layer in enumerate(self.layers):

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)

            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            if not skip_the_layer:
                # under deepspeed zero3 all gpus must run in sync
                # XXX: could optimize this like synced_gpus in generate_utils but not sure if it's worth the code complication
                if self.gradient_checkpointing and self.training:
                    # create gradient checkpointing function
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)
                        return custom_forward
                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer),
                        hidden_states
                    )
                else:
                    layer_outputs = layer(hidden_states)
                hidden_states = layer_outputs[0]


        hidden_states1 = self.layer_norm1(hidden_states[:,0,:,:])
        hidden_states2 = self.layer_norm2(hidden_states[:,1,:,:])
        # print(f'decoder end, hidden_states1 = {hidden_states1.shape}')
        hidden_states = torch.cat((hidden_states1.unsqueeze(dim=1), hidden_states2.unsqueeze(dim=1)), dim=1).mean(dim=1)
        # print(f'decoder end, hidden_states = {hidden_states.shape}')

        x = hidden_states
        # SED: multi-label multi-class classification, without sigmoid
        event_frame_logit = F.relu_(self.event_fc_1(self.event_dropout_1(x)))  # (batch_size, time_steps, n_classes)
        event_frame_logit = self.event_fc_2(self.event_dropout_2(event_frame_logit))
        # print(f'decoder end, sed = {event_frame_logit.shape}')
        # DOA: regression
        x_output = F.relu_(self.x_fc_1(self.x_dropout_1(x)))
        x_output = torch.tanh(self.x_fc_2(self.x_dropout_2(x_output)))
        y_output = F.relu_(self.y_fc_1(self.y_dropout_1(x)))
        y_output = torch.tanh(self.y_fc_2(self.y_dropout_2(y_output)))
        z_output = F.relu_(self.z_fc_1(self.z_dropout_1(x)))
        z_output = torch.tanh(self.z_fc_2(self.z_dropout_2(z_output)))
        doa_output = torch.cat((x_output, y_output, z_output), dim=-1)  # (batch_size, time_steps, 3 * n_classes)
        # print(f'decoder end, doa = {doa_output.shape}')
        output = {
            'event_frame_logit': event_frame_logit,
            'doa_frame_output': doa_output,
        }

        return output