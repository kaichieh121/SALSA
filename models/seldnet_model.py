import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
import os
from models import parameters
import torch.nn.functional as F
import shutil

from models.interfaces import BaseModel
from models.model_utils import interpolate_tensor


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)


    def forward(self, query, key, value, mask=None):

        batch_size = query.shape[0]

        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]

        energy = torch.div(torch.matmul(Q, K.permute(0, 1, 3, 2)), np.sqrt(self.head_dim))

        #energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim = -1)

        #attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)

        #x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        #x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        #x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        #x = [batch size, query len, hid dim]

        return x


class AttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, key_channels):
        super(AttentionLayer, self).__init__()
        self.conv_Q = nn.Conv1d(in_channels, key_channels, kernel_size=1, bias=False)
        self.conv_K = nn.Conv1d(in_channels, key_channels, kernel_size=1, bias=False)
        self.conv_V = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        Q = self.conv_Q(x)
        K = self.conv_K(x)
        V = self.conv_V(x)
        A = Q.permute(0, 2, 1).matmul(K).softmax(2)
        x = A.matmul(V.permute(0, 2, 1)).permute(0, 2, 1)
        return x

    def __repr__(self):
        return self._get_name() + \
            '(in_channels={}, out_channels={}, key_channels={})'.format(
            self.conv_Q.in_channels,
            self.conv_V.out_channels,
            self.conv_K.out_channels
            )


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):

        super().__init__()

        self.conv = torch.nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding)

        self.bn = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = torch.relu_(self.bn(self.conv(x)))
        return x


class SELDnet(BaseModel):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, sed_threshold: float = 0.3, doa_threshold: int = 20,
                 label_rate: int = 10, feature_rate: int = None, optimizer_name: str = 'Adam', lr: float = 1e-3,
                 loss_weight: Tuple = None, output_pred_dir: str = None, submission_dir: str = None,
                 test_chunk_len: int = None, test_chunk_hop_len: int = None, gt_meta_root_dir: str = None,
                 output_format: str = None, eval_version: str = '2021', is_eval: bool = False, **kwargs):
        super().__init__(sed_threshold=sed_threshold, doa_threshold=doa_threshold, label_rate=label_rate,
                         feature_rate=feature_rate, optimizer_name=optimizer_name, lr=lr,
                         output_pred_dir=output_pred_dir, submission_dir=submission_dir, test_chunk_len=test_chunk_len,
                         test_chunk_hop_len=test_chunk_hop_len, gt_meta_root_dir=gt_meta_root_dir,
                         output_format=output_format, eval_version=eval_version)

        task_id = '5'
        params = parameters.get_params(task_id)
        self.crnn = CRNN(in_feat_shape=(32, 7, 640, 200), out_shape=(32, 40, 36), params=params)


        self.save_hyperparameters()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_weight = loss_weight
        self.time_downsample_ratio = float(self.encoder.time_downsample_ratio)
        self.n_classes = self.decoder.n_classes
        self.doa_format = self.decoder.doa_format  # doa_format redundance since we have doa_output_format
        self.is_eval = is_eval

        self.seld_val = None

    def forward(self, x):
        """
        x: (batch_size, n_channels, n_timesteps (n_frames), n_features).
        """
        # output_dict = {
        #     'event_frame_logit': event_frame_logit, # (batch_size, n_timesteps, n_classes)
        #     'doa_frame_output': doa_output, # (batch_size, n_timesteps, 3* n_classes)
        # }
        output_dict = self.crnn(x)
        # print('output size:', output_dict['doa_frame_output'].shape)
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
        # interpolate output to match label rate
        pred_dict['event_frame_logit'] = interpolate_tensor(
            pred_dict['event_frame_logit'], ratio=5 * self.label_rate / self.feature_rate)
        pred_dict['doa_frame_output'] = interpolate_tensor(
            pred_dict['doa_frame_output'], ratio=5 * self.label_rate / self.feature_rate)
        # print(f'self.time_downsample_ratio = {self.time_downsample_ratio}')
        # print(f'self.label_rate = {self.label_rate}')
        # print(f'self.feature_rate = {self.feature_rate}')
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


class CRNN(nn.Module):
    def __init__(self, in_feat_shape, out_shape, params):
        super().__init__()
        self.nb_classes = params['unique_classes']
        self.conv_block_list = nn.ModuleList()
        if len(params['f_pool_size']):
            for conv_cnt in range(len(params['f_pool_size'])):
                self.conv_block_list.append(
                    ConvBlock(
                        in_channels=params['nb_cnn2d_filt'] if conv_cnt else in_feat_shape[1],
                        out_channels=params['nb_cnn2d_filt']
                    )
                )
                self.conv_block_list.append(
                    torch.nn.MaxPool2d((params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt]))
                )
                self.conv_block_list.append(
                    torch.nn.Dropout2d(p=params['dropout_rate'])
                )

        if params['nb_rnn_layers']:
            self.in_gru_size = params['nb_cnn2d_filt'] * int( np.floor(in_feat_shape[-1] / np.prod(params['f_pool_size'])))
            self.gru = torch.nn.GRU(input_size=self.in_gru_size, hidden_size=params['rnn_size'],
                                    num_layers=params['nb_rnn_layers'], batch_first=True,
                                    dropout=params['dropout_rate'], bidirectional=True)
        self.attn = None
        if params['self_attn']:
#            self.attn = AttentionLayer(params['rnn_size'], params['rnn_size'], params['rnn_size'])
            self.attn = MultiHeadAttentionLayer(params['rnn_size'], params['nb_heads'], params['dropout_rate'])

        self.fnn_list = torch.nn.ModuleList()
        if params['nb_rnn_layers'] and params['nb_fnn_layers']:
            for fc_cnt in range(params['nb_fnn_layers']):
                self.fnn_list.append(
                    torch.nn.Linear(params['fnn_size'] if fc_cnt else params['rnn_size'] , params['fnn_size'], bias=True)
                )
        self.fnn_list.append(
            torch.nn.Linear(params['fnn_size'] if params['nb_fnn_layers'] else params['rnn_size'], out_shape[-1], bias=True)
        )

        print(f'rnn_size = {params["rnn_size"]}')
        print(f'self.in_gru_size = {self.in_gru_size}')
        self.fc_size = params['rnn_size']
        # self.fc_size = self.in_gru_size
        self.event_fc_1 = nn.Linear(self.fc_size, self.fc_size // 2, bias=True)
        self.event_dropout_1 = nn.Dropout(p=0.2)
        self.event_fc_2 = nn.Linear(self.fc_size // 2, self.nb_classes, bias=True)
        self.event_dropout_2 = nn.Dropout(p=0.2)

    def forward(self, x):
        '''input: (batch_size, mic_channels, time_steps, mel_bins)'''

        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)
        '''(batch_size, feature_maps, time_steps, mel_bins)'''
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        ''' (batch_size, time_steps, feature_maps):'''

        (x, _) = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2]
        '''(batch_size, time_steps, feature_maps)'''
        if self.attn is not None:
            x = self.attn.forward(x, x, x)
            # out - batch x hidden x seq
            x = torch.tanh(x)

        x_rnn = x
        for fnn_cnt in range(len(self.fnn_list)-1):
            x = self.fnn_list[fnn_cnt](x)
        doa = torch.tanh(self.fnn_list[-1](x))
        '''(batch_size, time_steps, label_dim)'''

        event_frame_logit = F.relu_(self.event_fc_1(self.event_dropout_1(x_rnn)))  # (batch_size, time_steps, n_classes)
        event_frame_logit = self.event_fc_2(self.event_dropout_2(event_frame_logit))

        output = {
            'event_frame_logit': event_frame_logit,
            'doa_frame_output': doa,
        }

        return output
