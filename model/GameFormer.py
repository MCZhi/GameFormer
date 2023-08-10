import torch
from .modules import *


class Encoder(nn.Module):
    def __init__(self, neighbors_to_predict, layers=6):
        super(Encoder, self).__init__()
        heads, dim, dropout = 8, 256, 0.1
        self._neighbors = neighbors_to_predict
        self.agent_encoder = AgentEncoder()
        self.ego_encoder = AgentEncoder()
        self.lane_encoder = LaneEncoder()
        self.crosswalk_encoder = CrosswalkEncoder()
        attention_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4,
                                                     activation=F.gelu, dropout=dropout, batch_first=True)
        self.fusion_encoder = nn.TransformerEncoder(attention_layer, layers, enable_nested_tensor=False)

    def segment_map(self, map, map_encoding):
        stride = 10
        B, N_e, N_p, D = map_encoding.shape

        # segment map
        map_encoding = F.max_pool2d(map_encoding.permute(0, 3, 1, 2), kernel_size=(1, stride))
        map_encoding = map_encoding.permute(0, 2, 3, 1).reshape(B, -1, D)

        # segment mask
        map_mask = torch.eq(map, 0)[:, :, :, 0].reshape(B, N_e, N_p//stride, N_p//(N_p//stride))
        map_mask = torch.max(map_mask, dim=-1)[0].reshape(B, -1)

        return map_encoding, map_mask

    def forward(self, inputs):
        # agent encoding
        ego = inputs['ego_state']
        neighbors = inputs['neighbors_state']
        actors = torch.cat([inputs['ego_state'].unsqueeze(1), neighbors], dim=1)
        encoded_ego = self.ego_encoder(ego)
        encoded_neighbors = [self.agent_encoder(neighbors[:, i]) for i in range(neighbors.shape[1])]
        encoded_actors = torch.stack([encoded_ego] + encoded_neighbors, dim=1)
        actors_mask = torch.eq(actors[:, :, -1].sum(-1), 0)

        # map encoding
        map_lanes = inputs['map_lanes']
        map_crosswalks = inputs['map_crosswalks']
        encoded_map_lanes = self.lane_encoder(map_lanes)
        encoded_map_crosswalks = self.crosswalk_encoder(map_crosswalks)

        # attention fusion
        encodings = []
        masks = []
        N = self._neighbors + 1
        assert actors.shape[1] >= N, 'Too many neighbors to predict'

        for i in range(N):
            lanes, lanes_mask = self.segment_map(map_lanes[:, i], encoded_map_lanes[:, i])
            crosswalks, crosswalks_mask = self.segment_map(map_crosswalks[:, i], encoded_map_crosswalks[:, i])
            fusion_input = torch.cat([encoded_actors, lanes, crosswalks], dim=1)
            mask = torch.cat([actors_mask, lanes_mask, crosswalks_mask], dim=1)
            masks.append(mask)
            encoding = self.fusion_encoder(fusion_input, src_key_padding_mask=mask)
            encodings.append(encoding)

        # outputs
        encodings = torch.stack(encodings, dim=1)
        masks = torch.stack(masks, dim=1)
        encoder_outputs = {
            'actors': actors,
            'encodings': encodings,
            'masks': masks
        }

        return encoder_outputs


class Decoder(nn.Module):
    def __init__(self, modalities, future_len, neighbors_to_predict, levels=3):
        super(Decoder, self).__init__()
        self._levels = levels
        self._neighbors = neighbors_to_predict
        future_encoder = FutureEncoder()
        self.initial_stage = InitialDecoder(modalities, neighbors_to_predict, future_len)
        self.interaction_stage = nn.ModuleList([InteractionDecoder(future_encoder, future_len) for _ in range(levels)])  

    def forward(self, encoder_inputs):
        decoder_outputs = {}
        N = self._neighbors + 1
        assert encoder_inputs['actors'].shape[1] >= N, 'Too many neighbors to predict'

        current_states = encoder_inputs['actors'][:, :, -1]
        encodings, masks = encoder_inputs['encodings'], encoder_inputs['masks']

        # level 0
        results = [self.initial_stage(i, current_states[:, i], encodings[:, i], masks[:, i]) for i in range(N)]
        last_content = torch.stack([result[0] for result in results], dim=1)
        last_level = torch.stack([result[1] for result in results], dim=1)
        last_scores = torch.stack([result[2] for result in results], dim=1)
        decoder_outputs['level_0_interactions'] = last_level
        decoder_outputs['level_0_scores'] = last_scores
        
        # level k reasoning
        for k in range(1, self._levels+1):
            interaction_decoder = self.interaction_stage[k-1]
            results = [interaction_decoder(i, current_states[:, :N], last_level, last_scores, \
                       last_content[:, i], encodings[:, i], masks[:, i]) for i in range(N)]
            last_content = torch.stack([result[0] for result in results], dim=1)
            last_level = torch.stack([result[1] for result in results], dim=1)
            last_scores = torch.stack([result[2] for result in results], dim=1)
            decoder_outputs[f'level_{k}_interactions'] = last_level
            decoder_outputs[f'level_{k}_scores'] = last_scores

        return decoder_outputs


class GameFormer(nn.Module):
    def __init__(self, modalities, neighbors_to_predict, future_len, encoder_layers=6, decoder_levels=4):
        super(GameFormer, self).__init__()
        self.encoder = Encoder(neighbors_to_predict, encoder_layers)
        self.decoder = Decoder(modalities, future_len, neighbors_to_predict, decoder_levels)

    def forward(self, inputs):
        encoder_outputs = self.encoder(inputs)
        outputs = self.decoder(encoder_outputs)

        return outputs
