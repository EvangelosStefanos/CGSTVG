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


def modality_concatenation(self, feat_2d, feat_motion, feat_text):
    frame_length = feat_2d.size(0)
    feat_text = feat_text.expand(feat_text.size(0), frame_length, feat_text.size(-1))
    # concat visual and text features and Pad the vis_pos with 0 for the text tokens
    concat_features = torch.cat([feat_2d.permute(1,0,2), feat_text, feat_motion.permute(1,0,2)], dim=0)

    # TSNE START #
    if self.steps % 40338 < 10:
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
                    max_iter=300,
                )
                Y = tsne.fit_transform(X)
                ax.set_title("Perplexity=%d" % perplexities[i])
                ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2])
            fig.savefig(f"output/vidstg/tsne_{self.steps}.png")
            plt.close(fig)
    self.steps += 1
    # TSNE STOP #

    #vis_pos = torch.cat([pos_motion, torch.zeros_like(text_features), pos_rgb], dim=0)
    frames_cls = torch.mean(concat_features, dim=0)
    videos_cls = torch.mean(frames_cls, dim=0)
    pos_query, content_query = self.pos_fc(frames_cls), self.time_fc(videos_cls)
    pos_query = pos_query.sigmoid()  # [n_frames, bs, 4]
    content_query = content_query.expand(feat_2d.size(0), content_query.size(-1)).unsqueeze(1)  # [n_frames, bs, d_model]
    conf_query=self.conf(pos_query).sigmoid().squeeze()
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

        #self.vis_encoder = build_vis_encoder(cfg)
        #vis_fea_dim = self.vis_encoder.num_channels
      
        self.text_encoder = build_text_encoder(cfg)
        
        #self.ground_encoder = build_encoder(cfg)
        #self.ground_decoder = build_decoder(cfg)
        
        hidden_dim = cfg.MODEL.CG.HIDDEN
        #self.input_proj = nn.Conv2d(vis_fea_dim, hidden_dim, kernel_size=1)
        self.temp_embed = MLP(hidden_dim, hidden_dim, 2, 2, dropout=0.3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        #self.vid = vidswin_model("video_swin_t_p4w7", "video_swin_t_p4w7_k400_1k")
        #self.input_proj2 = nn.Conv2d(768, hidden_dim, kernel_size=1)
        #for param in self.vid.parameters():
        #    param.requires_grad = False

        self.action_embed = None
        if self.use_actioness:
            self.action_embed = MLP(hidden_dim, hidden_dim, 1, 2, dropout=0.3)

        #self.ground_decoder.time_embed2 = self.action_embed

        # add the iteration anchor update
        #self.ground_decoder.decoder.bbox_embed = self.bbox_embed
        
        #### V-JEPA extension ####
        self.vjepa_config = VJEPAConfig()
        if self.vjepa_config.use_bfloat16 == True:
            raise ValueError("bfloat16 is not supported.")
        self.vjepa_encoder = build_vjepa_encoder(self.vjepa_config)

        self.vjepa_classifier_motion = build_vjepa_classifier(
            config=self.vjepa_config,
            encoder=self.vjepa_encoder,
            video_data=True,
            checkpoint_path="model_zoo/vjepa/probes/k400-probe.pth.tar",
            frozen=True,
        )

        self.vjepa_classifier_2d = build_vjepa_classifier(
            config=self.vjepa_config,
            encoder=self.vjepa_encoder,
            video_data=False,
            checkpoint_path="model_zoo/vjepa/probes/in1k-probe.pth.tar",
            frozen=True,
        )

        self.NCLIPS = 8
        self.VIEWS_PER_CLIP = 1
        self.FRAMES_PER_SAMPLE = 128
        self.FRAMES_PER_CLIP = self.FRAMES_PER_SAMPLE // self.NCLIPS
        self.B = 1
        self.steps = 0

        ###embeds
        self.motion_embed = MLP(1, (self.NCLIPS * self.FRAMES_PER_CLIP) // 2, self.NCLIPS * self.FRAMES_PER_CLIP, 2,dropout=0.3)
        self.rgb_embed = MLP(1, (self.NCLIPS * self.FRAMES_PER_CLIP) // 2, self.NCLIPS * self.FRAMES_PER_CLIP, 2,dropout=0.3)

        self.mask_motion_embed = nn.Linear(self.vjepa_config.num_classes_vid, hidden_dim, bias=True)
        self.mask_rgb_embed = nn.Linear(self.vjepa_config.num_classes_img, hidden_dim, bias=True)

        ###Decoders####
        decoder_layer_2d = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=cfg.MODEL.CG.HEADS)
        self.decoder_2d = nn.TransformerDecoder(decoder_layer_2d, num_layers=cfg.MODEL.CG.DEC_LAYERS//3)

        decoder_layer_motion = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=cfg.MODEL.CG.HEADS)
        self.decoder_motion = nn.TransformerDecoder(decoder_layer_motion, num_layers=cfg.MODEL.CG.DEC_LAYERS//3)

        decoder_layer_text = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=cfg.MODEL.CG.HEADS)
        self.decoder_text = nn.TransformerDecoder(decoder_layer_text, num_layers=cfg.MODEL.CG.DEC_LAYERS//3)

        self.pos_fc = nn.Sequential(
            BertLayerNorm(256, eps=1e-12),
            nn.Dropout(0.1),
            nn.Linear(256, 4),
            nn.ReLU(True),
            BertLayerNorm(4, eps=1e-12),
        )

        self.time_fc = nn.Sequential(
            BertLayerNorm(256, eps=1e-12),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(True),
            BertLayerNorm(256, eps=1e-12),
        )

        self.conf = MLP(4, self.NCLIPS*self.FRAMES_PER_CLIP, 1, 3,  dropout=0.3)\

        self.d_model = self.cfg.MODEL.CG.HIDDEN
        if cfg.MODEL.CG.USE_LEARN_TIME_EMBED:
            self.tgt_embed = SeqEmbeddingLearned(self.NCLIPS * self.FRAMES_PER_CLIP + 1, self.d_model)
        else:
            self.tgt_embed = SeqEmbeddingSine(self.NCLIPS * self.FRAMES_PER_CLIP + 1, self.d_model)

        return



    def forward(self, videos, texts, targets, iteration_rate=-1):
        T, C, H, W = videos.tensors.shape # T = batch * clips * views_per_clip * frames_per_clip
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
            with torch.no_grad():
                vjepa_features = self.vjepa_encoder(clips, clip_indices)
                if self.vjepa_config.attend_across_segments:
                    outputs_motion = [self.vjepa_classifier_motion(o) for o in vjepa_features]
                    outputs_2d = [self.vjepa_classifier_2d(o) for o in vjepa_features]
                else:
                    outputs_motion = [[self.vjepa_classifier_motion(ost) for ost in os] for os in vjepa_features]
                    outputs_2d = [[self.vjepa_classifier_2d(ost) for ost in os] for os in vjepa_features]


            ###mask decoder features
            mask_motion=self.motion_embed(torch.permute(outputs_motion[0],(1,0)))
            mask_rgb = self.rgb_embed(torch.permute(outputs_2d[0],(1,0)))
            mask_motion = torch.unsqueeze(torch.permute(mask_motion, (1, 0)), 1)
            mask_rgb = torch.unsqueeze(torch.permute(mask_rgb, (1, 0)), 1)
            mask_motion = self.mask_motion_embed(mask_motion)
            mask_rgb = self.mask_rgb_embed(mask_rgb)


            ####positional embedding backbone
            position_embedding = build_position_encoding(self.cfg)
            ###mask position embeddings
            motion_pos=torch.unsqueeze(torch.permute(mask_motion, (0,2,1)),-1)
            rgb_pos = torch.unsqueeze(torch.permute(mask_rgb, (0,2,1)), -1)
            mask_pos=torch.unsqueeze(torch.zeros(motion_pos.size()[0],motion_pos.size()[2], dtype=torch.bool),-1).to(self.device)
            encoder_pos_motion = position_embedding(motion_pos, mask_pos)
            encoder_pos_rgb = position_embedding(rgb_pos, mask_pos)
            encoder_pos_motion=torch.squeeze(torch.permute(encoder_pos_motion,(0,2,1,3)),3)
            encoder_pos_rgb = torch.squeeze(torch.permute(encoder_pos_rgb, (0, 2, 1, 3)), 3)


            ####tgt input and positional encoding
            tgt = torch.zeros(self.FRAMES_PER_SAMPLE, self.B, self.d_model).to(self.device)
            tgt_pos = self.tgt_embed(self.FRAMES_PER_SAMPLE).to(self.device)

            ##decoder masks
            tgt_mask_visual=nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(self.device)
            if nframes_required>0:
                memory_key_padding_mask = torch.ones(self.B, tgt.size(0)).byte().to(self.device)
                memory_key_padding_mask[:,:tgt.size(0)-nframes_required]=False
            else:
                memory_key_padding_mask = torch.zeros(self.B, tgt.size(0)).byte().to(self.device)

            ###visual decoding
            output_motion_padded=self.decoder_motion(tgt+tgt_pos,mask_motion+encoder_pos_motion, tgt_mask=tgt_mask_visual, memory_key_padding_mask=memory_key_padding_mask.bool())
            output_2d_padded = self.decoder_2d(tgt+tgt_pos, mask_rgb+encoder_pos_rgb, tgt_mask=tgt_mask_visual)

            ###keep unpadded tensors
            output_motion=output_motion_padded[:tgt.size(0)-nframes_required,:,:]
            output_2d = output_2d_padded[:tgt.size(0) - nframes_required, :, :]


            # Textual Feature
            device = clips.device
            text_outputs, _ = self.text_encoder(texts, device)
            mask_text=text_outputs[1]

            # expand the attention mask and text token from [b, len] to [n_frames, len]
              # [text_len, n_frames, d_model]


            #text position embeddings
            text_pos=torch.unsqueeze(torch.permute(text_outputs[1], (2, 1, 0)), 0)
            mask_pos_t=torch.unsqueeze(torch.zeros(text_pos.size()[0],text_pos.size()[3], dtype=torch.bool),1).to(self.device)
            encoder_pos_text = position_embedding(text_pos, mask_pos_t)
            encoder_pos_text = torch.squeeze(torch.permute(encoder_pos_text, (3, 0, 1, 2)),-1)

            ####tgt input and positional encoding
            tgt_text = torch.zeros(mask_text.size()[0], self.B, self.d_model).to(self.device)
            tgt_pos_text = self.tgt_embed(mask_text.size()[0]).to(self.device)

            ##decoder masks
            tgt_mask_textual = nn.Transformer.generate_square_subsequent_mask(tgt_text.size(0)).to(self.device)

            ##textual decoder
            output_text = self.decoder_text(tgt_text+tgt_pos_text,mask_text+encoder_pos_text, tgt_mask=tgt_mask_textual)

            pos_query, time_query, conf_query = modality_concatenation(self, output_2d, output_motion, output_text)
            
            NUM_LAYERS = 1
            pos_query = pos_query.reshape(shape=(NUM_LAYERS, pos_query.size(0), 4)) # [FRAMES_PER_SAMPLE, 4] -> [NUM_LAYERS, FRAMES_PER_SAMPLE, 4]
            conf_query = conf_query.reshape(shape=(NUM_LAYERS, conf_query.size(0))) # [FRAMES_PER_SAMPLE] -> [NUM_LAYERS, FRAMES_PER_SAMPLE]

            out = {}

            # the final decoder embeddings and the refer anchors
            ###############  predict bounding box ###############

            #outputs_coord = refer_anchors.flatten(1,2)  # [num_layers, T, 4]
            out.update({"pred_boxes": pos_query[-1]})
            out.update({"boxes_conf": conf_query[-1]})
            ######################################################

            #######  predict the start and end probability #######
            time_hiden_state = time_query
            outputs_time = self.temp_embed(time_hiden_state)  # [num_layers, b, T, 2]
            outputs_time = outputs_time.permute(dims=(1,0,2))
            outputs_time = outputs_time.reshape(shape=(NUM_LAYERS, self.B, time_query.size(0), 2)) # [B, FRAMES_PER_SAMPLE, 2] -> [NUM_LAYERS, B, FRAMES_PER_SAMPLE, 2]
            out.update({"pred_sted": outputs_time[-1]})
            #######################################################

            if self.use_actioness:
                outputs_actioness = self.action_embed(time_hiden_state).reshape(shape=(-1, self.B, time_query.size(0), 1))  # [num_layers, b, FRAMES_PER_SAMPLE, 1]
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
