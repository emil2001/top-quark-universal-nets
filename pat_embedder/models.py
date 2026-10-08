import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import detector_enc_padding_mask


class Block(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.BatchNorm1d(out_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.need_proj = (in_dim != out_dim)
        if self.need_proj:
            self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        residual = self.proj(x) if self.need_proj else x
        out = self.dropout(self.activation(self.norm(self.linear(x))))
        return out + residual


def _make_head(in_dim, out_dim, dropout):
    mid = max(in_dim // 2, out_dim)
    return nn.Sequential(
        nn.Linear(in_dim, mid),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(mid, out_dim),
    )


# ---------------------------------------------------------------------------
# V1 heads
# ---------------------------------------------------------------------------

class RegressionHeads(nn.Module):
    def __init__(self, hidden_dim, max_leptons, max_jets, max_tops, dropout=0.1):
        super().__init__()
        self.lepton = _make_head(hidden_dim, max_leptons * 7, dropout)
        self.neutrino = _make_head(hidden_dim, max_leptons * 7, dropout)
        self.jet = _make_head(hidden_dim, max_jets * 7, dropout)
        self.top_quark = _make_head(hidden_dim, max_tops * 7, dropout)
        self.w = _make_head(hidden_dim, max_tops * 7, dropout)

    def forward(self, features, batch_size, max_leptons, max_jets, max_tops):
        return {
            'lepton': self.lepton(features).view(batch_size, max_leptons, 7),
            'neutrino': self.neutrino(features).view(batch_size, max_leptons, 7),
            'jet': self.jet(features).view(batch_size, max_jets, 7),
            'top_quark': self.top_quark(features).view(batch_size, max_tops, 7),
            'w': self.w(features).view(batch_size, max_tops, 7),
        }


class ClassificationHeads(nn.Module):
    def __init__(self, hidden_dim, max_leptons, max_jets, max_tops, dropout=0.1):
        super().__init__()
        self.lepton = _make_head(hidden_dim, max_leptons * 7, dropout)
        self.neutrino = _make_head(hidden_dim, max_leptons * 7, dropout)
        self.jet = _make_head(hidden_dim, max_jets * 3, dropout)
        self.top_quark = _make_head(hidden_dim, max_tops * 3, dropout)
        self.w = _make_head(hidden_dim, max_tops * 3, dropout)

    def forward(self, features, batch_size, max_leptons, max_jets, max_tops):
        return {
            'lepton': self.lepton(features).view(batch_size, max_leptons, 7),
            'neutrino': self.neutrino(features).view(batch_size, max_leptons, 7),
            'jet': self.jet(features).view(batch_size, max_jets, 3),
            'top_quark': self.top_quark(features).view(batch_size, max_tops, 3),
            'w': self.w(features).view(batch_size, max_tops, 3),
        }


class AssignmentHeads(nn.Module):
    def __init__(self, hidden_dim, max_leptons, max_jets, max_tops, max_ws, dropout=0.1):
        super().__init__()
        self.max_tops = max_tops
        self.max_ws = max_ws
        self.top_to_w = _make_head(hidden_dim, max_tops * (max_ws + 1), dropout)
        self.top_to_bjet = _make_head(hidden_dim, max_tops * (max_jets + 1), dropout)
        self.w_to_lepton = _make_head(hidden_dim, max_ws * (max_leptons + 1), dropout)
        self.w_to_neutrino = _make_head(hidden_dim, max_ws * (max_leptons + 1), dropout)
        self.w_to_jet1 = _make_head(hidden_dim, max_ws * (max_jets + 1), dropout)
        self.w_to_jet2 = _make_head(hidden_dim, max_ws * (max_jets + 1), dropout)

    def forward(self, features, batch_size):
        return {
            'top_to_w':       self.top_to_w(features).view(batch_size, self.max_tops, -1),
            'top_to_bjet':    self.top_to_bjet(features).view(batch_size, self.max_tops, -1),
            'w_to_lepton':    self.w_to_lepton(features).view(batch_size, self.max_ws, -1),
            'w_to_neutrino':  self.w_to_neutrino(features).view(batch_size, self.max_ws, -1),
            'w_to_jet1':      self.w_to_jet1(features).view(batch_size, self.max_ws, -1),
            'w_to_jet2':      self.w_to_jet2(features).view(batch_size, self.max_ws, -1),
        }


class HighLevelRegressionHead(nn.Module):
    def __init__(self, hidden_dim, num_hlr_vars, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_hlr_vars),
        )

    def forward(self, features):
        return self.head(features)


# ---------------------------------------------------------------------------
# V1 model (MLP backbone)
# ---------------------------------------------------------------------------

class PatEmbedderModel(nn.Module):
    def __init__(
        self,
        max_leptons=4,
        max_jets=12,
        max_tops=4,
        max_ws=4,
        hidden_dim=256,
        num_layers=4,
        dropout=0.2,
        num_hlr_vars=0,
    ):
        super().__init__()
        self.max_leptons = max_leptons
        self.max_jets = max_jets
        self.max_tops = max_tops
        self.max_ws = max_ws
        self.num_hlr_vars = num_hlr_vars

        met_dim = 2
        lepton_dim = 8
        jet_dim = 8
        self.input_dim = met_dim + max_leptons * lepton_dim + max_jets * jet_dim

        in_dim = self.input_dim
        extractor_blocks = []
        for _ in range(num_layers):
            extractor_blocks.append(Block(in_dim, hidden_dim, dropout))
            in_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*extractor_blocks)

        head_dropout = dropout * 0.5
        self.reg_heads = RegressionHeads(hidden_dim, max_leptons, max_jets, max_tops, head_dropout)
        self.cls_heads = ClassificationHeads(hidden_dim, max_leptons, max_jets, max_tops, head_dropout)
        self.assign_heads = AssignmentHeads(hidden_dim, max_leptons, max_jets, max_tops, max_ws, head_dropout)
        self.hlr_head = (
            HighLevelRegressionHead(hidden_dim, num_hlr_vars, head_dropout)
            if num_hlr_vars > 0 else None
        )

    def forward(self, x):
        batch_size = x.size(0)
        features = self.feature_extractor(x)
        reg = self.reg_heads(features, batch_size, self.max_leptons, self.max_jets, self.max_tops)
        cls = self.cls_heads(features, batch_size, self.max_leptons, self.max_jets, self.max_tops)
        assign = self.assign_heads(features, batch_size)
        hlr = self.hlr_head(features) if self.hlr_head is not None else None

        return (
            reg['lepton'],
            reg['neutrino'],
            reg['jet'],
            reg['top_quark'],
            reg['w'],
            cls['lepton'],
            cls['neutrino'],
            cls['jet'],
            cls['top_quark'],
            cls['w'],
            assign,
            hlr,
        )


class BilinearAssignHead(nn.Module):
    """Assignment via scaled dot-product between query and key embeddings.
    Appends a learned 'no-match' key so output has N_keys + 1 columns."""

    def __init__(self, embed_dim, n_keys):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.no_match_key = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.scale = embed_dim ** -0.5

    def forward(self, queries, keys):
        B, N_q, D = queries.shape
        Q = self.query_proj(queries)
        K = self.key_proj(keys)
        no_match = self.no_match_key.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        K = torch.cat([K, no_match], dim=1)
        return torch.bmm(Q, K.transpose(1, 2)) * self.scale


# ---------------------------------------------------------------------------
# V3 model (DETR-style encoder-decoder)
# ---------------------------------------------------------------------------

REG_DIM = 4      # (pt, eta, phi, E)  — used by V3/V4-style targets
REG_DIM_V5 = 5   # (pt, eta, sin_phi, cos_phi, E) — used by V5


class MultiHeadAssignHead(nn.Module):
    """Multi-head cross-attention assignment with learned no-match score.

    Returns logits of shape (B, N_queries, N_keys + 1) where last column is "none".
    """

    def __init__(self, embed_dim, n_keys, n_heads=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )
        self.no_match_score = nn.Parameter(torch.zeros(1))
        self.n_keys = n_keys

    def forward(self, queries, keys):
        B, N_q, D = queries.shape
        N_k = keys.shape[1]
        ctx, _ = self.cross_attn(queries, keys, keys)
        ctx = self.norm(ctx + queries)
        q_exp = ctx.unsqueeze(2).expand(B, N_q, N_k, D)
        k_exp = keys.unsqueeze(1).expand(B, N_q, N_k, D)
        pair_feat = torch.cat([q_exp, k_exp], dim=-1)
        scores = self.scorer(pair_feat).squeeze(-1)
        no_match = self.no_match_score.expand(B, N_q, 1)
        return torch.cat([scores, no_match], dim=-1)


class PatEmbedderModelV3(nn.Module):
    """Encoder-decoder with deep supervision + presence logits.

    Input: same flat (B, 130) as other versions.
    Outputs: (lep_reg, neu_reg, jet_reg, top_reg, w_reg,
              lep_cls, neu_cls, jet_cls, top_cls, w_cls,
              assign_dict, hlr, aux_head_outputs, dn_outputs)
    """

    def __init__(
        self,
        max_leptons=4,
        max_jets=12,
        max_tops=4,
        max_ws=4,
        embed_dim=256,
        n_heads=4,
        n_enc_layers=6,
        n_dec_layers=4,
        dropout=0.1,
        num_hlr_vars=0,
        reg_dim=REG_DIM,
    ):
        super().__init__()
        self.max_leptons = max_leptons
        self.max_jets = max_jets
        self.max_tops = max_tops
        self.max_ws = max_ws
        self.num_hlr_vars = num_hlr_vars
        self.embed_dim = embed_dim
        self.reg_dim = reg_dim

        met_dim = 2
        lepton_dim = 8
        jet_dim = 8
        self.input_dim = met_dim + max_leptons * lepton_dim + max_jets * jet_dim

        self.met_encoder = nn.Sequential(
            nn.Linear(met_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.lepton_encoder = nn.Sequential(
            nn.Linear(lepton_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.jet_encoder = nn.Sequential(
            nn.Linear(jet_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Type embeddings: MET=0, lepton=1, jet=2, top=3, W=4
        self.type_embedding = nn.Embedding(5, embed_dim)

        # Within-category positional embeddings
        self.lepton_pos = nn.Parameter(torch.randn(max_leptons, embed_dim) * 0.02)
        self.jet_pos = nn.Parameter(torch.randn(max_jets, embed_dim) * 0.02)

        # Learnable padding tokens (substituted for absent particles)
        self.lepton_pad = nn.Parameter(torch.zeros(embed_dim))
        self.jet_pad = nn.Parameter(torch.zeros(embed_dim))

        # Learnable query tokens for decoder
        self.top_queries = nn.Parameter(torch.randn(max_tops, embed_dim) * 0.02)
        self.w_queries = nn.Parameter(torch.randn(max_ws, embed_dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_enc_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec_layers)

        hd = dropout * 0.5
        self.lepton_reg_head = _make_head(embed_dim, reg_dim, hd)
        self.neutrino_reg_head = _make_head(embed_dim * 2, reg_dim, hd)
        self.jet_reg_head = _make_head(embed_dim, reg_dim, hd)
        self.top_reg_head = _make_head(embed_dim, reg_dim, hd)
        self.w_reg_head = _make_head(embed_dim, reg_dim, hd)

        self.lepton_cls_head = _make_head(embed_dim, 7, hd)
        self.neutrino_cls_head = _make_head(embed_dim * 2, 7, hd)
        self.jet_cls_head = _make_head(embed_dim, 3, hd)
        self.top_cls_head = _make_head(embed_dim, 3, hd)
        self.w_cls_head = _make_head(embed_dim, 3, hd)

        self.assign_top_to_w = MultiHeadAssignHead(embed_dim, max_ws, n_heads=n_heads)
        self.assign_top_to_bjet = MultiHeadAssignHead(embed_dim, max_jets, n_heads=n_heads)
        self.assign_w_to_lepton = MultiHeadAssignHead(embed_dim, max_leptons, n_heads=n_heads)
        self.assign_w_to_neutrino = MultiHeadAssignHead(embed_dim, max_leptons, n_heads=n_heads)
        self.assign_w_to_jet1 = MultiHeadAssignHead(embed_dim, max_jets, n_heads=n_heads)
        self.assign_w_to_jet2 = MultiHeadAssignHead(embed_dim, max_jets, n_heads=n_heads)

        self.top_presence_head = nn.Linear(embed_dim, 1)
        self.w_presence_head = nn.Linear(embed_dim, 1)

        if num_hlr_vars > 0:
            self.hlr_pool_query = nn.Parameter(torch.randn(1, embed_dim) * 0.02)
            self.hlr_head = _make_head(embed_dim, num_hlr_vars, hd)
        else:
            self.hlr_pool_query = None
            self.hlr_head = None

        # DN-DETR denoising: project GT regression to query space
        self.dn_content_enc = nn.Sequential(
            nn.Linear(reg_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.dn_label_enc = nn.Embedding(6, embed_dim)  # 3 top + 3 W class labels

        # Encoder token layout: [MET(1) | leptons | jets]
        self._enc_n = 1 + max_leptons + max_jets
        self._lep_slice = slice(1, 1 + max_leptons)
        self._jet_slice = slice(1 + max_leptons, self._enc_n)
        # Decoder query layout: [tops | Ws]
        self._dec_n = max_tops + max_ws
        self._top_q_slice = slice(0, max_tops)
        self._w_q_slice = slice(max_tops, max_tops + max_ws)

    def _apply_head_batched(self, head, tokens, n):
        B = tokens.size(0)
        return head(tokens.reshape(B * n, -1)).view(B, n, -1)

    def _build_enc_padding_mask(self, x):
        """Detector-derived encoder padding mask from the model input features.

        Delegates to the single shared source ``detector_enc_padding_mask``: a
        slot is padding iff its detector feature block is all-zero. The same
        mask is built explicitly on the training/evaluation paths, so this
        in-model fallback (used when a caller omits an explicit mask) is
        identical to what training uses — no train/eval skew, no parton-truth.
        """
        return detector_enc_padding_mask(x, self.max_leptons, self.max_jets)

    def forward(self, x, enc_padding_mask=None, dn_reg=None, dn_cls=None, compute_aux=True):
        """Forward pass.

        Args:
            x: (B, input_dim) flat features.
            enc_padding_mask: optional (B, 1 + max_leptons + max_jets) bool tensor
                where True = ignored (padding). When None, the detector-derived
                mask is built from ``x`` via ``_build_enc_padding_mask`` (the same
                source used by the training/eval paths), so the in-model fallback
                matches training. Callers may also supply it explicitly.
            dn_reg: optional (B, n_dn, reg_dim) noised GT regression for DN-DETR.
            dn_cls: optional (B, n_dn) DN class indices into dn_label_enc. Top
                labels live at indices 0..2; W labels are offset by max_top_classes
                (indices 3..5) — callers must apply this offset.
            compute_aux: whether to materialize intermediate decoder-head outputs
                for deep supervision.
        """
        B, device = x.size(0), x.device

        met_feat = x[:, :2]
        lep_feat = x[:, 2:2 + self.max_leptons * 8].view(B, self.max_leptons, 8)
        jet_feat = x[:, 2 + self.max_leptons * 8:].view(B, self.max_jets, 8)

        met_emb_raw = self.met_encoder(met_feat).unsqueeze(1)
        lep_emb = self.lepton_encoder(lep_feat.reshape(B * self.max_leptons, 8)).view(B, self.max_leptons, -1)
        jet_emb = self.jet_encoder(jet_feat.reshape(B * self.max_jets, 8)).view(B, self.max_jets, -1)

        if enc_padding_mask is None:
            enc_padding_mask = self._build_enc_padding_mask(x)

        if enc_padding_mask is not None:
            # Slot is "present" where the mask is False.
            lep_present = ~enc_padding_mask[:, self._lep_slice]
            jet_present = ~enc_padding_mask[:, self._jet_slice]
            lep_emb = torch.where(
                lep_present.unsqueeze(-1), lep_emb, self.lepton_pad.expand_as(lep_emb)
            )
            jet_emb = torch.where(
                jet_present.unsqueeze(-1), jet_emb, self.jet_pad.expand_as(jet_emb)
            )

        tid = torch.arange(5, device=device)
        met_emb = met_emb_raw + self.type_embedding(tid[0])
        lep_emb = lep_emb + self.type_embedding(tid[1]) + self.lepton_pos
        jet_emb = jet_emb + self.type_embedding(tid[2]) + self.jet_pos

        enc_tokens = torch.cat([met_emb, lep_emb, jet_emb], dim=1)
        enc_out = self.encoder(enc_tokens, src_key_padding_mask=enc_padding_mask)

        top_q = self.top_queries.unsqueeze(0).expand(B, -1, -1) + self.type_embedding(tid[3])
        w_q = self.w_queries.unsqueeze(0).expand(B, -1, -1) + self.type_embedding(tid[4])
        dec_queries = torch.cat([top_q, w_q], dim=1)

        n_dn = 0
        dn_attn_mask = None
        if dn_reg is not None and dn_cls is not None:
            dn_content = self.dn_content_enc(dn_reg)
            dn_label = self.dn_label_enc(dn_cls)
            dn_q = dn_content + dn_label
            n_dn = dn_q.shape[1]
            dec_queries = torch.cat([dec_queries, dn_q], dim=1)

            n_match = self._dec_n
            n_total = n_match + n_dn
            dn_attn_mask = torch.zeros(n_total, n_total, device=device, dtype=torch.bool)
            dn_attn_mask[:n_match, n_match:] = True
            dn_attn_mask[n_match:, :n_match] = True

        dec_hidden = dec_queries
        aux_outputs = []
        for layer_idx, dec_layer in enumerate(self.decoder.layers):
            dec_hidden = dec_layer(
                dec_hidden,
                enc_out,
                tgt_mask=dn_attn_mask,
                memory_key_padding_mask=enc_padding_mask,
            )
            if compute_aux and layer_idx < len(self.decoder.layers) - 1:
                aux_top = dec_hidden[:, self._top_q_slice, :]
                aux_w = dec_hidden[:, self._w_q_slice, :]
                aux_outputs.append((aux_top, aux_w))
        dec_out = dec_hidden

        lep_enc = enc_out[:, self._lep_slice, :]
        jet_enc = enc_out[:, self._jet_slice, :]
        top_dec = dec_out[:, self._top_q_slice, :]
        w_dec = dec_out[:, self._w_q_slice, :]

        # Use the encoder-processed MET token (carries context from all leptons + jets)
        # rather than the raw projected MET; gives the neutrino head richer evidence.
        met_enc = enc_out[:, 0:1, :].expand(B, self.max_leptons, -1)
        lep_met = torch.cat([lep_enc, met_enc], dim=-1)

        lep_reg = self._apply_head_batched(self.lepton_reg_head, lep_enc, self.max_leptons)
        neu_reg = self._apply_head_batched(self.neutrino_reg_head, lep_met, self.max_leptons)
        jet_reg = self._apply_head_batched(self.jet_reg_head, jet_enc, self.max_jets)
        top_reg = self._apply_head_batched(self.top_reg_head, top_dec, self.max_tops)
        w_reg = self._apply_head_batched(self.w_reg_head, w_dec, self.max_ws)

        lep_cls = self._apply_head_batched(self.lepton_cls_head, lep_enc, self.max_leptons)
        neu_cls = self._apply_head_batched(self.neutrino_cls_head, lep_met, self.max_leptons)
        jet_cls = self._apply_head_batched(self.jet_cls_head, jet_enc, self.max_jets)
        top_cls = self._apply_head_batched(self.top_cls_head, top_dec, self.max_tops)
        w_cls = self._apply_head_batched(self.w_cls_head, w_dec, self.max_ws)

        assign = {
            "top_to_w": self.assign_top_to_w(top_dec, w_dec),
            "top_to_bjet": self.assign_top_to_bjet(top_dec, jet_enc),
            "w_to_lepton": self.assign_w_to_lepton(w_dec, lep_enc),
            "w_to_neutrino": self.assign_w_to_neutrino(w_dec, lep_enc),
            "w_to_jet1": self.assign_w_to_jet1(w_dec, jet_enc),
            "w_to_jet2": self.assign_w_to_jet2(w_dec, jet_enc),
        }
        assign["top_presence"] = self.top_presence_head(top_dec).squeeze(-1)
        assign["w_presence"] = self.w_presence_head(w_dec).squeeze(-1)

        aux_head_outputs = []
        if compute_aux:
            for aux_top_h, aux_w_h in aux_outputs:
                aux_head_outputs.append(
                    {
                        "top_reg": self._apply_head_batched(self.top_reg_head, aux_top_h, self.max_tops),
                        "w_reg": self._apply_head_batched(self.w_reg_head, aux_w_h, self.max_ws),
                        "top_cls": self._apply_head_batched(self.top_cls_head, aux_top_h, self.max_tops),
                        "w_cls": self._apply_head_batched(self.w_cls_head, aux_w_h, self.max_ws),
                        "assign": {
                            "top_to_w": self.assign_top_to_w(aux_top_h, aux_w_h),
                            "top_to_bjet": self.assign_top_to_bjet(aux_top_h, jet_enc),
                            "w_to_lepton": self.assign_w_to_lepton(aux_w_h, lep_enc),
                            "w_to_neutrino": self.assign_w_to_neutrino(aux_w_h, lep_enc),
                            "w_to_jet1": self.assign_w_to_jet1(aux_w_h, jet_enc),
                            "w_to_jet2": self.assign_w_to_jet2(aux_w_h, jet_enc),
                        },
                    }
                )

        hlr = None
        if self.hlr_head is not None:
            all_tok = torch.cat([enc_out, dec_out[:, :self._dec_n, :]], dim=1)
            q = self.hlr_pool_query.expand(B, -1, -1)
            attn = torch.bmm(q, all_tok.transpose(1, 2)) / math.sqrt(self.embed_dim)
            if enc_padding_mask is not None:
                full_mask = torch.cat(
                    [
                        enc_padding_mask,
                        torch.zeros(B, self._dec_n, dtype=torch.bool, device=device),
                    ],
                    dim=1,
                )
                attn = attn.masked_fill(full_mask.unsqueeze(1), float("-inf"))
            pooled = torch.bmm(F.softmax(attn, dim=-1), all_tok).squeeze(1)
            hlr = self.hlr_head(pooled)

        dn_outputs = None
        if n_dn > 0:
            dn_dec = dec_out[:, self._dec_n:, :]
            n_dn_tops = self.max_tops
            n_dn_ws = self.max_ws
            dn_top_h = dn_dec[:, :n_dn_tops, :]
            dn_w_h = dn_dec[:, n_dn_tops:n_dn_tops + n_dn_ws, :]
            dn_outputs = {
                "top_reg": self._apply_head_batched(self.top_reg_head, dn_top_h, n_dn_tops),
                "w_reg": self._apply_head_batched(self.w_reg_head, dn_w_h, n_dn_ws),
                "top_cls": self._apply_head_batched(self.top_cls_head, dn_top_h, n_dn_tops),
                "w_cls": self._apply_head_batched(self.w_cls_head, dn_w_h, n_dn_ws),
            }

        return (
            lep_reg,
            neu_reg,
            jet_reg,
            top_reg,
            w_reg,
            lep_cls,
            neu_cls,
            jet_cls,
            top_cls,
            w_cls,
            assign,
            hlr,
            aux_head_outputs,
            dn_outputs,
        )
