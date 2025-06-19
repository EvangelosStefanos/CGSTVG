from torch import nn
from torch.nn.functional import dropout

from .net_utils import MLP
from .vision_model import build_vis_encoder
from .language_model import build_text_encoder
from .grounding_model import build_encoder, build_decoder
from utils.misc import NestedTensor
from .vidswin.video_swin_transformer import vidswin_model

from new.vjepa import build_vjepa_encoder, build_vjepa_classifier, vjepa_predict, VJEPAConfig
import torch
from models.vision_model.position_encoding import build_position_encoding
from models.grounding_model.position_encoding import SeqEmbeddingLearned, SeqEmbeddingSine
from models.bert_model.bert_module import BertLayerNorm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils.comm import is_main_process

class EvCapAttention(nn.Module):
    def __init__(self, v_dim, elem_dim, hidden_dim, attn_dim):
        super().__init__()
        # Element-aware attention
        self.W_v = nn.Linear(v_dim, attn_dim)
        self.W_p = nn.Linear(elem_dim, attn_dim)
        self.W_s = nn.Linear(attn_dim, 1)

        # Decoder-guided attention
        self.W_h = nn.Linear(hidden_dim, attn_dim)
        self.W_v_prime = nn.Linear(v_dim + elem_dim, attn_dim)
        self.W_d = nn.Linear(attn_dim, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, V, P, H):
        """
        Args:
            V: visual features (B, R, v_dim)
            P: element features (B, elem_dim)
            H: decoder hidden state (B, hidden_dim)
        Returns:
            V_hat: attended context vector (B, v_dim + elem_dim)
        """
        B, R, _ = V.shape

        # === Step 1: Element-aware attention ===
        V_proj = self.W_v(V)                          # (B, R, attn_dim)
        P_proj = self.W_p(P).unsqueeze(1)             # (B, 1, attn_dim)
        alpha = self.W_s*(self.sigmoid(V_proj + P_proj)).squeeze(-1)  # (B, R)
        #alpha = F.softmax(alpha, dim=1)
        V_att = torch.bmm(alpha.unsqueeze(1), V).squeeze(1)  # (B, v_dim)
        V_prime = torch.cat([V_att, P], dim=-1)             # (B, v_dim + elem_dim)

        # === Step 2: Decoder-guided attention ===
        # Repeat V′ across regions (optional — depends on use case)
        V_prime_seq = V_prime.unsqueeze(1).repeat(1, R, 1)  # (B, R, v_dim + elem_dim)
        H_expanded = H.unsqueeze(1).expand(-1, R, -1)       # (B, R, hidden_dim)

        h_proj = self.W_h(H_expanded)
        v_proj = self.W_v_prime(V_prime_seq)
        beta = self.W_d*(self.sigmoid(h_proj + v_proj)).squeeze(-1)
        #beta = F.softmax(beta, dim=1)

        V_hat = torch.bmm(beta.unsqueeze(1), V_prime_seq).squeeze(1)  # (B, v_dim + elem_dim)

        return V_hat

    def forward(self, H, V_prime):
        """
        Args:
            H: Decoder hidden state, shape (batch_size, hidden_dim)
            V_prime: Attended visual features, shape (batch_size, num_regions, v_dim)
        Returns:
            V_hat: Final attended visual vector, shape (batch_size, v_dim)
        """
        batch_size, num_regions, v_dim = V_prime.shape

        # Expand H to (batch_size, num_regions, hidden_dim)
        H_expanded = H.unsqueeze(1).expand(-1, num_regions, -1)

        # Linear projections
        h_proj = self.W_h(H_expanded)        # (batch_size, num_regions, attn_dim)
        v_proj = self.W_v(V_prime)           # (batch_size, num_regions, attn_dim)

        # Combine and apply sigmoid
        combined = self.sigmoid(h_proj + v_proj)  # (batch_size, num_regions, attn_dim)

        # Project to scalar attention weights
        beta = self.W_d(combined).squeeze(-1)     # (batch_size, num_regions)

        # Normalize with softmax
        alpha = F.softmax(beta, dim=1)            # (batch_size, num_regions)

        # Weighted sum of V_prime
        V_hat = torch.bmm(alpha.unsqueeze(1), V_prime).squeeze(1)  # (batch_size, v_dim)

        return V_hat




def modality_concatenation(self, feat_2d, feat_motion, feat_text, feat_temporal):
    T, B, E = feat_2d.shape
    W = feat_text.shape[0]
    #plot means-stds #
    STEPS_PER_EPOCH = 40338 # 2 gpus
    if is_main_process() and self.steps % STEPS_PER_EPOCH < 10:
        with torch.no_grad():
            (fig, subplots) = plt.subplots(1, 2, figsize=(19.2, 10.8), layout="constrained", squeeze=False)
            feats = [feat_2d, feat_motion, feat_text]
            colors = ["r", "g", "b"]
            labels = ["image", "text", "motion"]
            for i, ax in enumerate(subplots.flat):
                if i==0:
                    ax.set_title("Mean-Stds")
                    for j in range(len(feats)):
                        feats[j] = feats[j].detach().cpu()
                        (std, mean) = torch.std_mean(feats[j], dim=0)
                        mean = mean.flatten()
                        std = std.flatten()
                        x = list(range(len(mean)))
                        # ax.scatter(x, mean, s=100*std, alpha=0.5, c=colors[j], label=labels[j])
                        ax.errorbar(x=x, y=mean, yerr=std, alpha=0.5, linestyle="None", c=colors[j], label=labels[j], fmt="-o")
                    ax.legend()
                elif i==1:
                    ax.set_title("Mean-Stds-Total")
                    for j in range(len(feats)):
                        feats[j] = feats[j].detach().cpu()
                        (std, mean) = torch.std_mean(feats[j])
                        mean = mean.flatten()
                        std = std.flatten()
                        # ax.scatter(x, mean, s=100*std, alpha=0.5, c=colors[j], label=labels[j])
                        ax.errorbar(x=j, y=mean, yerr=std, alpha=0.5, linestyle="None", c=colors[j], label=labels[j], fmt="-o")
                    ax.legend()
            fig.savefig(f"{self.cfg.OUTPUT_DIR}mean-std_{self.steps}.png")
            plt.close(fig)
    
    frame_length = feat_2d.size(0)
    feat_text = feat_text.expand(feat_text.size(0), frame_length, feat_text.size(-1))

    feat_2d=feat_2d.permute(1,0,2)
    feat_motion=feat_motion.permute(1,0,2)

    if self.cfg.MODEL.NORMALIZATION=='clamp':
        # clamp here if needed
        feat_2d = feat_2d.clamp(min=-1,max=1)
        feat_motion = feat_motion.clamp(min=-1,max=1)
        feat_text = feat_text.clamp(min=-1,max=1)
        feat_temporal = feat_temporal.clamp(min=-1,max=1)
    elif self.cfg.MODEL.NORMALIZATION=='layer':
        #normalization layer
        feat_2d = self.layer_norm_2d(feat_2d)
        feat_motion = self.layer_norm_motion(feat_motion)
        feat_text = self.layer_norm_2d(feat_text)
    
    # concat visual and text features and Pad the vis_pos with 0 for the text tokens
    concat_features = torch.cat([feat_2d, feat_text, feat_motion], dim=0)

    # TSNE START #
    STEPS_PER_EPOCH = 40338 # 2 gpus
    if is_main_process() and self.steps % STEPS_PER_EPOCH < 10:
        with torch.no_grad():
            W, T, E = feat_text.shape
            X = concat_features.reshape(shape=((W+2)*T, E)).detach().cpu()
            (fig, subplots) = plt.subplots(2, 2, figsize=(19.2, 10.8), subplot_kw=dict(projection='3d'), layout="constrained")
            perplexities = [5, 30, 50, 100]
            for i, ax in enumerate(subplots.flat):
                tsne = TSNE(
                    n_components=3,
                    init="random",
                    random_state=0,
                    perplexity=perplexities[i],
                    n_iter=300,
                )
                Y = tsne.fit_transform(X)
                ax.set_title("Perplexity=%d" % perplexities[i])

                p = Y[0*T:1*T]
                ax.scatter(p[:, 0], p[:, 1], p[:, 2], c="r", label="image")

                p = Y[1*T:13*T]
                ax.scatter(p[:, 0], p[:, 1], p[:, 2], c="g", label="text")

                p = Y[13*T:14*T]
                ax.scatter(p[:, 0], p[:, 1], p[:, 2], c="b", label="motion")

                ax.legend()
            fig.savefig(f"{self.cfg.OUTPUT_DIR}tsne_{self.steps}.png")
            plt.close(fig)
    self.steps += 1
    # TSNE STOP #

    #post fusion decoding # [(W+2)*Τ, Β, Ε] -> [(W+2)*Τ, Β, Ε]
    TT = (W+2)*T
    tgt = torch.zeros(TT, B, E).to(self.device) # [TT, B, E]
    tgt_pos = self.post_fusion_tgt_embed(TT).to(self.device) # [TT, B, E]
    mask_motion = concat_features.reshape((-1, B, E)) # [W+2, T, E] >> [TT, 1, E]
    motion_pos = torch.unsqueeze(torch.permute(mask_motion, (0, 2, 1)), -1) # [TT, B, E] >> [TT, E, B, 1]
    mask_pos = torch.unsqueeze(torch.zeros(motion_pos.size()[0], motion_pos.size()[2], dtype=torch.bool), -1).to(self.device) # [TT, 1, 1]
    encoder_pos_motion = torch.squeeze(torch.permute(self.position_embedding(motion_pos, mask_pos), (0, 2, 1, 3)), 3) # [TT, E, B, 1] >> [TT, B, E]
    # frames_cls = self.post_fusion_decoder(
    #     tgt=tgt+tgt_pos,
    #     tgt_mask=nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(self.device),
    #     memory=mask_motion + encoder_pos_motion,
    # ).reshape((W+2, T, E))


    #vis_pos = torch.cat([pos_motion, torch.zeros_like(text_features), pos_rgb], dim=0)
    frames_cls = torch.mean(concat_features, dim=0)

    if self.cfg.MODEL.TEMPORAL_BRANCH == 'a':
        videos_cls=torch.mean(feat_temporal, dim=0).squeeze()
    else:
        videos_cls = torch.mean(frames_cls, dim=0)


    pos_query, content_query = self.pos_fc(frames_cls), self.time_fc(videos_cls)
    pos_query = pos_query.sigmoid()  # [n_frames, bs, 4]
    content_query = content_query.expand(feat_2d.size(1), content_query.size(-1)).unsqueeze(
        1)  # [n_frames, bs, d_model]
    conf_query = self.conf(pos_query).sigmoid().squeeze()
    return pos_query, content_query, conf_query


class CGSTVG(nn.Module):
    def __init__(self, cfg):
        super(CGSTVG, self).__init__()
        self.cfg = cfg.clone()
        self.max_video_len = cfg.INPUT.MAX_VIDEO_LEN
        self.use_attn = cfg.SOLVER.USE_ATTN
        
        self.use_aux_loss = cfg.SOLVER.USE_AUX_LOSS  # use the output of each transformer layer
        self.use_actioness = cfg.MODEL.CG.USE_ACTION
        self.query_dim = cfg.MODEL.CG.QUERY_DIM
        DROPOUT = self.cfg.MODEL.CG.DROPOUT

        # self.vis_encoder = build_vis_encoder(cfg)
        # vis_fea_dim = self.vis_encoder.num_channels

        self.text_encoder = build_text_encoder(cfg)

        # self.ground_encoder = build_encoder(cfg)
        # self.ground_decoder = build_decoder(cfg)

        hidden_dim = cfg.MODEL.CG.HIDDEN
        # self.input_proj = nn.Conv2d(vis_fea_dim, hidden_dim, kernel_size=1)
        self.temp_embed = MLP(hidden_dim, hidden_dim, 2, 3, dropout=DROPOUT)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # self.vid = vidswin_model("video_swin_t_p4w7", "video_swin_t_p4w7_k400_1k")
        # self.input_proj2 = nn.Conv2d(768, hidden_dim, kernel_size=1)
        # for param in self.vid.parameters():
        #    param.requires_grad = False

        self.NCLIPS = 8
        self.VIEWS_PER_CLIP = 1
        self.FRAMES_PER_SAMPLE = 128
        self.FRAMES_PER_CLIP = self.FRAMES_PER_SAMPLE // self.NCLIPS
        self.B = 1
        self.steps = 0

        self.action_embed = None
        self.vjepa_config = VJEPAConfig()

        if self.use_actioness:
            self.action_embed = MLP(hidden_dim, hidden_dim, 1, 3, dropout=DROPOUT)

        # self.ground_decoder.time_embed2 = self.action_embed

        # add the iteration anchor update
        # self.ground_decoder.decoder.bbox_embed = self.bbox_embed

        if self.cfg.MODEL.CGSTVG_ENCODERS:
            self.vid = vidswin_model("video_swin_t_p4w7", "video_swin_t_p4w7_k400_1k")
            self.vis_encoder = build_vis_encoder(cfg)

            vis_fea_dim = self.vis_encoder.num_channels
            self.input_proj = nn.Conv2d(vis_fea_dim, hidden_dim, kernel_size=1)
            self.input_proj2 = nn.Conv2d(768, hidden_dim, kernel_size=1)
        else:

            #### V-JEPA extension ####

            if self.vjepa_config.use_bfloat16 == True:
                raise ValueError("bfloat16 is not supported.")
            self.vjepa_encoder = build_vjepa_encoder(self.vjepa_config)
            self.FROZEN = True

            if self.cfg.MODEL.FRAME_DIMENSION == True:
                frames_number = self.FRAMES_PER_SAMPLE
            else:
                frames_number=1

            self.vjepa_classifier_motion = build_vjepa_classifier(
                config=self.vjepa_config,
                encoder=self.vjepa_encoder,
                video_data=True,
                checkpoint_path="model_zoo/vjepa/probes/k400-probe.pth.tar",
                frozen=self.FROZEN,
                frames_number=frames_number,
            )

            self.vjepa_classifier_2d = build_vjepa_classifier(
                config=self.vjepa_config,
                encoder=self.vjepa_encoder,
                video_data=False,
                checkpoint_path="model_zoo/vjepa/probes/in1k-probe.pth.tar",
                frozen=self.FROZEN,
                frames_number=frames_number,
            )

            if self.cfg.MODEL.TEMPORAL_BRANCH=='a':
                self.vjepa_classifier_temporal = build_vjepa_classifier(
                    config=self.vjepa_config,
                    encoder=self.vjepa_encoder,
                    video_data=True,
                    checkpoint_path="model_zoo/vjepa/probes/k400-probe.pth.tar",
                    frozen=self.FROZEN,
                    frames_number=frames_number,
                )

        if self.cfg.MODEL.FRAME_DIMENSION == False:
            ###embeds
            self.motion_embed = MLP(1, (self.FRAMES_PER_SAMPLE) // 2, self.FRAMES_PER_SAMPLE, 3, dropout=DROPOUT)
            self.rgb_embed = MLP(1, (self.FRAMES_PER_SAMPLE) // 2, self.FRAMES_PER_SAMPLE, 3, dropout=DROPOUT)

        self.mask_motion_embed = nn.Linear(self.vjepa_config.num_classes_vid, hidden_dim, bias=True)
        self.mask_rgb_embed = nn.Linear(self.vjepa_config.num_classes_img, hidden_dim, bias=True)

        ###Decoders####
        decoder_layer_2d = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=cfg.MODEL.CG.HEADS)
        self.decoder_2d = nn.TransformerDecoder(decoder_layer_2d, num_layers=cfg.MODEL.CG.DEC_LAYERS)

        decoder_layer_motion = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=cfg.MODEL.CG.HEADS)
        self.decoder_motion = nn.TransformerDecoder(decoder_layer_motion, num_layers=cfg.MODEL.CG.DEC_LAYERS)

        decoder_layer_text = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=cfg.MODEL.CG.HEADS)
        self.decoder_text = nn.TransformerDecoder(decoder_layer_text, num_layers=cfg.MODEL.CG.DEC_LAYERS)

        max_sentence_length = 30
        self.post_fusion_tgt_embed = SeqEmbeddingSine(max_sentence_length * self.FRAMES_PER_SAMPLE + 1, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=cfg.MODEL.CG.HEADS)
        self.post_fusion_decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.MODEL.CG.DEC_LAYERS)

        ##temporal decoder
        if self.cfg.MODEL.TEMPORAL_BRANCH == 'a':
            ##temporal embed
            if self.cfg.MODEL.FRAME_DIMENSION == False:
                self.temporal_embed = MLP(1, (self.FRAMES_PER_SAMPLE) // 2, self.FRAMES_PER_SAMPLE, 3, dropout=DROPOUT)
            # temporal mask #
            self.mask_temporal_embed = nn.Linear(self.vjepa_config.num_classes_vid, hidden_dim, bias=True)
            # temporal decoder #
            decoder_layer_temporal = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=cfg.MODEL.CG.HEADS)
            self.decoder_temporal = nn.TransformerDecoder(decoder_layer_temporal, num_layers=cfg.MODEL.CG.DEC_LAYERS)


        self.d_model = self.cfg.MODEL.CG.HIDDEN
        self.pos_fc = nn.Sequential(
            BertLayerNorm(self.d_model, eps=1e-12),
            nn.Dropout(DROPOUT),
            nn.Linear(self.d_model, 4),
            nn.ReLU(True),
            BertLayerNorm(4, eps=1e-12),
        )

        self.time_fc = nn.Sequential(
            BertLayerNorm(self.d_model, eps=1e-12),
            nn.Dropout(DROPOUT),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(True),
            BertLayerNorm(self.d_model, eps=1e-12),
        )

        self.conf = MLP(4, self.FRAMES_PER_SAMPLE, 1, 3, dropout=DROPOUT)

        # self.d_model = self.cfg.MODEL.CG.HIDDEN
        if cfg.MODEL.CG.USE_LEARN_TIME_EMBED:
            self.tgt_embed = SeqEmbeddingLearned(self.FRAMES_PER_SAMPLE + 1, self.d_model)
        else:
            self.tgt_embed = SeqEmbeddingSine(self.FRAMES_PER_SAMPLE + 1, self.d_model)

        ####positional embedding backbone
        self.position_embedding = build_position_encoding(self.cfg)

        if self.cfg.MODEL.NORMALIZATION == 'layer':

            self.layer_norm_2d = nn.LayerNorm(self.d_model)
            self.layer_norm_motion = nn.LayerNorm(self.d_model)
            self.layer_norm_text = nn.LayerNorm(self.d_model)






        return

    def forward(self, videos, texts, targets, iteration_rate=-1):


        T, C, H, W = videos.tensors.shape  # T = batch * clips * views_per_clip * frames_per_clip
        frame_ids = torch.tensor(targets[0]["frame_ids"])

        nframes_required = self.FRAMES_PER_SAMPLE - T
        if nframes_required > 0:
            pad = videos.tensors[-1].unsqueeze(0).repeat(nframes_required, 1, 1, 1)
            videos.tensors = torch.cat(tensors=(videos.tensors, pad), dim=0)

            pad = frame_ids[-1].unsqueeze(0).repeat(nframes_required)
            frame_ids = torch.cat(tensors=(frame_ids, pad), dim=0)

        clips = videos.tensors.reshape(shape=(self.B, self.NCLIPS, self.VIEWS_PER_CLIP, self.FRAMES_PER_CLIP, C, H, W))
        clips = clips.permute(dims=(1, 2, 0, 4, 3, 5, 6)) # [B, CLIPS, VIEWS, FRAMES_PER_CLIP, C, H, W] -> [CLIPS, VIEWS, B, C, FRAMES_PER_CLIP, H, W]
        clip_indices = torch.reshape(frame_ids, (self.NCLIPS, self.FRAMES_PER_CLIP))


        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=self.vjepa_config.use_bfloat16):

            if self.cfg.MODEL.CGSTVG_ENCODERS:

                ##2d backbone
                vis_outputs, vis_pos_embed = self.vis_encoder(videos)
                vis_features, vis_mask, vis_durations = vis_outputs.decompose()
                vis_features=torch.mean(vis_features,dim=[2,3], keepdim=True)
                outputs_2d = [torch.squeeze(self.input_proj(vis_features),dim=-1)]

                ###3d backbone
                vid_features = self.vid(videos.tensors, len(videos.tensors))
                outputs_motion=torch.mean(vid_features['3'],dim=[2,3], keepdim=True)
                outputs_motion=[torch.squeeze(self.input_proj2(outputs_motion),dim=-1)]

            else:

                with torch.no_grad():
                    vjepa_features = self.vjepa_encoder(clips, clip_indices)

                if self.FROZEN:
                    with torch.no_grad():
                        if self.vjepa_config.attend_across_segments:
                            outputs_motion = [self.vjepa_classifier_motion(o) for o in vjepa_features]
                            outputs_2d = [self.vjepa_classifier_2d(o) for o in vjepa_features]
                            if self.cfg.MODEL.TEMPORAL_BRANCH == 'a':
                                outputs_temporal = [self.vjepa_classifier_temporal(o) for o in vjepa_features]
                        else:
                            outputs_motion = [[self.vjepa_classifier_motion(ost) for ost in os] for os in
                                              vjepa_features]
                            outputs_2d = [[self.vjepa_classifier_2d(ost) for ost in os] for os in vjepa_features]
                            if self.cfg.MODEL.TEMPORAL_BRANCH == 'a':
                                outputs_temporal = [[self.vjepa_classifier_temporal(ost) for ost in os] for os in
                                                    vjepa_features]
                else:
                    if self.vjepa_config.attend_across_segments:
                        outputs_motion = [self.vjepa_classifier_motion(o) for o in vjepa_features]
                        outputs_2d = [self.vjepa_classifier_2d(o) for o in vjepa_features]
                        if self.cfg.MODEL.TEMPORAL_BRANCH == 'a':
                            outputs_temporal = [self.vjepa_classifier_temporal(o) for o in vjepa_features]
                    else:
                        outputs_motion = [[self.vjepa_classifier_motion(ost) for ost in os] for os in vjepa_features]
                        outputs_2d = [[self.vjepa_classifier_2d(ost) for ost in os] for os in vjepa_features]
                        if self.cfg.MODEL.TEMPORAL_BRANCH == 'a':
                            outputs_temporal = [[self.vjepa_classifier_temporal(ost) for ost in os] for os in
                                                vjepa_features]





            ###mask decoder features
            if self.cfg.MODEL.FRAME_DIMENSION == False:
                mask_motion = self.motion_embed(torch.permute(outputs_motion[0], (1, 0)))
                mask_rgb = self.rgb_embed(torch.permute(outputs_2d[0], (1, 0)))
                mask_motion = torch.unsqueeze(torch.permute(mask_motion, (1, 0)), 1)
                mask_rgb = torch.unsqueeze(torch.permute(mask_rgb, (1, 0)), 1)
            else:
                if self.cfg.MODEL.CGSTVG_ENCODERS:
                    mask_motion = torch.permute(outputs_motion[0], (0,2,1))
                    mask_rgb = torch.permute(outputs_2d[0], (0,2,1))
                else:
                    mask_motion = torch.permute(outputs_motion[0], (1, 0, 2))
                    mask_rgb = torch.permute(outputs_2d[0], (1, 0, 2))

            if self.cfg.MODEL.CGSTVG_ENCODERS==False:
                mask_motion = self.mask_motion_embed(mask_motion)
                mask_rgb = self.mask_rgb_embed(mask_rgb)

            ###mask position embeddings
            motion_pos = torch.unsqueeze(torch.permute(mask_motion, (0, 2, 1)), -1)
            rgb_pos = torch.unsqueeze(torch.permute(mask_rgb, (0, 2, 1)), -1)
            mask_pos = torch.unsqueeze(torch.zeros(motion_pos.size()[0], motion_pos.size()[2], dtype=torch.bool), -1).to(self.device)
            encoder_pos_motion = self.position_embedding(motion_pos, mask_pos)
            encoder_pos_rgb = self.position_embedding(rgb_pos, mask_pos)
            encoder_pos_motion = torch.squeeze(torch.permute(encoder_pos_motion, (0, 2, 1, 3)), 3)
            encoder_pos_rgb = torch.squeeze(torch.permute(encoder_pos_rgb, (0, 2, 1, 3)), 3)

            ####tgt input and positional encoding
            tgt = torch.zeros(self.FRAMES_PER_SAMPLE, self.B, self.d_model).to(self.device)
            tgt_pos = self.tgt_embed(self.FRAMES_PER_SAMPLE).to(self.device)

            ##decoder masks
            tgt_mask_visual = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(self.device)
            if nframes_required > 0:
                memory_key_padding_mask = torch.ones(self.B, tgt.size(0)).byte().to(self.device)
                memory_key_padding_mask[:, :tgt.size(0) - nframes_required] = False
            else:
                memory_key_padding_mask = torch.zeros(self.B, tgt.size(0)).byte().to(self.device)

            ##visual decoding
            output_motion_padded = self.decoder_motion(
                tgt + tgt_pos,
                mask_motion + encoder_pos_motion,
                tgt_mask=tgt_mask_visual,
                memory_key_padding_mask=memory_key_padding_mask.bool()
            )

            output_2d_padded = self.decoder_2d(
                tgt + tgt_pos,
                mask_rgb + encoder_pos_rgb,
                tgt_mask=tgt_mask_visual,
                memory_key_padding_mask=memory_key_padding_mask.bool()
            )

            ###keep unpadded tensors
            output_motion = output_motion_padded[:tgt.size(0) - nframes_required, :, :]
            output_2d = output_2d_padded[:tgt.size(0) - nframes_required, :, :]

            # Textual Feature
            device = clips.device
            text_outputs, _ = self.text_encoder(texts, device)
            mask_text = text_outputs[1]

            # expand the attention mask and text token from [b, len] to [n_frames, len]
            # [text_len, n_frames, d_model]

            # text position embeddings
            text_pos = torch.unsqueeze(torch.permute(text_outputs[1], (2, 1, 0)), 0)
            mask_pos_t = torch.unsqueeze(torch.zeros(text_pos.size()[0], text_pos.size()[3], dtype=torch.bool), 1).to(self.device)
            encoder_pos_text = self.position_embedding(text_pos, mask_pos_t)
            encoder_pos_text = torch.squeeze(torch.permute(encoder_pos_text, (3, 0, 1, 2)), -1)

            ####tgt input and positional encoding
            tgt_text = torch.zeros(mask_text.size()[0], self.B, self.d_model).to(self.device)
            tgt_pos_text = self.tgt_embed(mask_text.size()[0]).to(self.device)

            ##decoder masks
            tgt_mask_textual = nn.Transformer.generate_square_subsequent_mask(tgt_text.size(0)).to(self.device)

            ##textual decoder
            output_text = self.decoder_text(tgt_text + tgt_pos_text, mask_text + encoder_pos_text, tgt_mask=tgt_mask_textual)

            ##temporal decoding
            output_temporal = 0
            if self.cfg.MODEL.TEMPORAL_BRANCH == 'a':
                if self.cfg.MODEL.FRAME_DIMENSION == False:
                    mask_temporal = self.temporal_embed(torch.permute(outputs_temporal[0], (1, 0)))
                    mask_temporal = torch.unsqueeze(torch.permute(mask_temporal, (1, 0)), 1)
                else:
                    mask_temporal = torch.permute(outputs_temporal[0], (1, 0, 2))

                mask_temporal = self.mask_motion_embed(mask_temporal)
                temporal_pos = torch.unsqueeze(torch.permute(mask_temporal, (0, 2, 1)), -1)
                encoder_pos_temporal = self.position_embedding(temporal_pos, mask_pos)

                encoder_pos_temporal = torch.squeeze(torch.permute(encoder_pos_temporal, (0, 2, 1, 3)), 3)
                output_temporal_padded = self.decoder_temporal(
                    tgt + tgt_pos,
                    mask_temporal + encoder_pos_temporal,
                    tgt_mask=tgt_mask_visual,
                    memory_key_padding_mask=memory_key_padding_mask.bool()
                )

                output_temporal = output_temporal_padded[:tgt.size(0) - nframes_required, :, :]

            pos_query, time_query, conf_query = modality_concatenation(self, output_2d, output_motion, output_text, output_temporal)


            NUM_LAYERS = 1
            pos_query = pos_query.reshape(shape=(NUM_LAYERS, pos_query.size(0), 4))  # [FRAMES_PER_SAMPLE, 4] -> [NUM_LAYERS, FRAMES_PER_SAMPLE, 4]
            conf_query = conf_query.reshape(shape=(NUM_LAYERS, conf_query.size(0)))  # [FRAMES_PER_SAMPLE] -> [NUM_LAYERS, FRAMES_PER_SAMPLE]

            out = {}

            # the final decoder embeddings and the refer anchors
            ###############  predict bounding box ###############

            # outputs_coord = refer_anchors.flatten(1,2)  # [num_layers, T, 4]
            out.update({"pred_boxes": pos_query[-1]})
            out.update({"boxes_conf": conf_query[-1]})
            ######################################################

            #######  predict the start and end probability #######
            time_hiden_state = time_query
            outputs_time = self.temp_embed(time_hiden_state)  # [num_layers, b, T, 2]
            outputs_time = outputs_time.permute(dims=(1, 0, 2))
            outputs_time = outputs_time.reshape(shape=(NUM_LAYERS, self.B, time_query.size(0), 2))  # [B, FRAMES_PER_SAMPLE, 2] -> [NUM_LAYERS, B, FRAMES_PER_SAMPLE, 2]
            out.update({"pred_sted": outputs_time[-1]})
            #######################################################

            if self.use_actioness:
                outputs_actioness = self.action_embed(time_hiden_state).reshape(
                    shape=(-1, self.B, time_query.size(0), 1))  # [num_layers, b, FRAMES_PER_SAMPLE, 1]
                out.update({"pred_actioness": outputs_actioness[-1]})

            if self.use_aux_loss:
                out["aux_outputs"] = [
                    {
                        "pred_sted": a,
                        "pred_boxes": b,
                        "boxes_conf": c
                    }
                    for a, b, c in zip(outputs_time[:-1], pos_query[:-1], conf_query[:-1])
                ]
                for i_aux in range(len(out["aux_outputs"])):
                    if self.use_actioness:
                        out["aux_outputs"][i_aux]["pred_actioness"] = outputs_actioness[i_aux]

        return out