import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.beats.BEATs import BEATsModel
import torchaudio
from transformers import ClapModel, ClapProcessor
from transformers import (
    Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model,
    WavLMModel, WavLMConfig,
    AutoModel, AutoFeatureExtractor,
)


class XLSR(nn.Module):
    def __init__(
        self,
        model_dir,
        device="cuda",
        sampling_rate=16000,
        freeze=True,
        visual=False,
        return_hidden_states=False,
        selected_layers=None,
        layer_fusion="last",  # "last", "cat_linear", "cat_proj_v1", "cat_proj_v2", "mean", "weight_sum"
    ):
        super(XLSR, self).__init__()

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sampling_rate = sampling_rate
        self.return_hidden_states = return_hidden_states
        self.selected_layers = (
            tuple(int(i) for i in selected_layers) if selected_layers is not None else None
        )

        self.config = Wav2Vec2Config.from_json_file(f"{model_dir}/config.json")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_dir,
            do_normalize=False,
        )
        self.model = Wav2Vec2Model.from_pretrained(model_dir).to(self.device)
        self.freeze = freeze

        # 必须打开，否则 outputs.hidden_states 会是 None
        self.model.config.output_hidden_states = True

        self.visual = visual

        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.model.train()

        self.hidden_size = self.model.config.hidden_size  # XLSR-300M 通常是 1024

        lf = str(layer_fusion).strip().lower()
        if lf == "cat":
            lf = "cat_proj_v2"
        elif lf == "cat_proj":
            lf = "cat_proj_v2"
        self.layer_fusion = lf
        if lf in (
            "cat_linear",
            "cat_proj_v1",
            "cat_proj_v2",
            "mean",
            "weight_sum",
        ):
            if not self.selected_layers:
                raise ValueError(
                    "XLSR: selected_layers must be a non-empty sequence when "
                    f"layer_fusion={lf!r}"
                )
        elif lf == "last" and self.selected_layers is not None:
            if len(self.selected_layers) > 1:
                raise ValueError(
                    "XLSR: layer_fusion='last' accepts at most one "
                    f"selected_layers index; got {self.selected_layers!r}"
                )

        n_sel = len(self.selected_layers) if self.selected_layers else 0
        self.per_layer_ln = None
        self.layer_proj = None
        self.cat_linear_head = None
        self.layer_weights = None

        if lf == "cat_linear" and n_sel > 0:
            self.cat_linear_head = nn.Linear(
                self.hidden_size * n_sel, self.hidden_size
            )
        elif lf == "cat_proj_v1" and n_sel > 0:
            self.layer_proj = nn.Sequential(
                nn.LayerNorm(self.hidden_size * n_sel),
                nn.Linear(self.hidden_size * n_sel, self.hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
            )
        elif lf == "cat_proj_v2" and n_sel > 0:
            self.per_layer_ln = nn.ModuleList(
                [nn.LayerNorm(self.hidden_size) for _ in range(n_sel)]
            )
            self.layer_proj = nn.Sequential(
                nn.LayerNorm(self.hidden_size * n_sel),
                nn.Linear(self.hidden_size * n_sel, self.hidden_size),
                nn.Dropout(0.1),
            )
        elif lf == "weight_sum" and n_sel > 0:
            self.layer_weights = nn.Parameter(torch.zeros(n_sel))

    def _fuse_hidden_states(self, outputs):
        last_hidden_state = outputs.last_hidden_state
        hidden_states = outputs.hidden_states
        n_h = len(hidden_states)

        # "last": default final layer, OR a single explicit index in selected_layers.
        # Do not return last_hidden_state when user picked one intermediate layer.
        if self.layer_fusion == "last":
            if not self.selected_layers:
                return last_hidden_state, hidden_states
            if len(self.selected_layers) > 1:
                raise ValueError(
                    "XLSR: layer_fusion='last' accepts at most one index in "
                    "selected_layers; for multiple layers use cat_linear, "
                    "cat_proj_v1, cat_proj_v2, "
                    f"mean, or weight_sum. Got selected_layers={self.selected_layers!r}."
                )
            idx = self.selected_layers[0]
            if idx < 0 or idx >= n_h:
                raise IndexError(
                    f"selected_layers index {idx} out of range for "
                    f"{n_h} hidden_states (valid 0..{n_h - 1})."
                )
            return hidden_states[idx], hidden_states

        if not self.selected_layers:
            return last_hidden_state, hidden_states

        for i in self.selected_layers:
            if i < 0 or i >= n_h:
                raise IndexError(
                    f"selected_layers index {i} out of range for "
                    f"{n_h} hidden_states (valid 0..{n_h - 1})."
                )

        selected = [hidden_states[i] for i in self.selected_layers]

        if self.layer_fusion == "cat_linear":
            # Direct concat [B,T,C*N] -> Linear -> [B,T,C]
            fused = torch.cat(selected, dim=-1)
            fused = self.cat_linear_head(fused)

        elif self.layer_fusion == "cat_proj_v1":
            # Concat [B,T,C*N] -> LN+Linear+GELU+Dropout -> [B,T,C]
            fused = torch.cat(selected, dim=-1)
            fused = self.layer_proj(fused)

        elif self.layer_fusion == "cat_proj_v2":
            # Per-layer LN -> concat [B,T,C*N] -> project -> [B,T,C]
            normed = [
                ln(h) for ln, h in zip(self.per_layer_ln, selected)
            ]
            fused = torch.cat(normed, dim=-1)
            fused = self.layer_proj(fused)

        elif self.layer_fusion == "mean":
            # [B, T, C] * N -> [B, T, C]
            fused = torch.stack(selected, dim=0).mean(dim=0)

        elif self.layer_fusion == "weight_sum":
            weights = F.softmax(self.layer_weights, dim=0)
            fused = sum(w * h for w, h in zip(weights, selected))

        else:
            raise ValueError(
                f"Unsupported layer_fusion={self.layer_fusion}. "
                "Choose from ['last', 'cat_linear', 'cat_proj_v1', 'cat_proj_v2', "
                "'mean', 'weight_sum']."
            )

        return fused, hidden_states

    def forward(self, audio_data):
        feat = self.processor(
            audio_data,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        ).input_values.to(self.device)

        feat = feat.squeeze(dim=0)

        if self.visual:
            outputs = self.model(
                feat,
                output_attentions=True,
                output_hidden_states=True,
            )
            attentions = outputs.attentions
            fused, hidden_states = self._fuse_hidden_states(outputs)

            if self.return_hidden_states:
                return fused, attentions, hidden_states
            return fused, attentions

        if self.freeze:
            with torch.no_grad():
                outputs = self.model(
                    feat,
                    output_hidden_states=True,
                )
                fused, hidden_states = self._fuse_hidden_states(outputs)
        else:
            outputs = self.model(
                feat,
                output_hidden_states=True,
            )
            fused, hidden_states = self._fuse_hidden_states(outputs)

        if self.return_hidden_states:
            return fused, hidden_states

        return fused

    def extract_features(self, audio_data):
        return self.forward(audio_data)
    

class WAVLM(nn.Module):
    # WAVLM-Large output dimension is 1024
    def __init__(self, model_dir, device='cuda', sampling_rate=16000, freeze=True):
        super(WAVLM, self).__init__()
        # Set device (GPU or CPU)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sampling_rate = sampling_rate

        # Load the pre-trained model configuration and weights
        self.config = WavLMConfig.from_json_file(f"{model_dir}/config.json")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir, do_normalize = False)
        self.model = WavLMModel.from_pretrained(model_dir).to(self.device)
        self.freeze = freeze
        # Enable output of hidden states
        self.model.config.output_hidden_states = True

        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.model.train()

    def forward(self, audio_data):
        # Process the input audio using Wav2Vec2 Feature Extractor
        feat = self.processor(audio_data, sampling_rate=self.sampling_rate, return_tensors="pt").input_values.to(self.device)
        feat = feat.squeeze(dim=0)  
        if self.freeze:
            with torch.no_grad():
                output = self.model(feat).last_hidden_state
        else:
            output = self.model(feat).last_hidden_state
        return output
    
    def extract_features(self, audio_data):
        # Process the input audio and extract the features using the forward pass
        return self.forward(audio_data)  # Return the final layer's output


class MERT(nn.Module):
    def __init__(self, model_dir, device='cuda', sampling_rate=16000, freeze=True):
        # MERT-v1-330M output dimension is 1024
        super(MERT, self).__init__()

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sampling_rate = sampling_rate

        # Load the pre-trained model configuration and weights
        self.config = Wav2Vec2Config.from_json_file(f"{model_dir}/config.json")
        self.processor = AutoFeatureExtractor.from_pretrained(model_dir, sampling_rate = 16000,do_normalize = False)
        self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(self.device)
        self.freeze = freeze

        # Enable output of hidden states
        self.model.config.output_hidden_states = True
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.model.train()
            
    def forward(self, audio_data):
        # Process the input audio using Wav2Vec2 Feature Extractor
        feat = self.processor(audio_data, sampling_rate=self.sampling_rate, return_tensors="pt").input_values.to(self.device)
        feat = feat.squeeze(dim=0)  
        if self.freeze:
            with torch.no_grad():
                output = self.model(feat).last_hidden_state
        else:
            output = self.model(feat).last_hidden_state
        return output
    
    def extract_features(self, audio_data):
        # Process the input audio and extract the features using the forward pass
        return self.forward(audio_data)  # Return the final layer's output


class PT_XLSR(nn.Module):
    def __init__(self, model_dir, prompt_dim, device='cuda', sampling_rate=16000, num_prompt_tokens=10, dropout=0.1, visual=False):
        super(PT_XLSR, self).__init__()

        # Set device (GPU or CPU)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sampling_rate = sampling_rate

        # Load the pre-trained model configuration and weights
        self.config = Wav2Vec2Config.from_json_file(f"{model_dir}/config.json")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir, do_normalize = False)
        self.model = Wav2Vec2Model.from_pretrained(model_dir).to(self.device)

        # Enable output of hidden states
        self.model.config.output_hidden_states = True
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Create a learnable prompt embedding for 24 layers
        self.prompt_dim = prompt_dim
        self.num_prompt_tokens = num_prompt_tokens  # Assume prompt consists of 10 tokens
        self.prompt_embedding = nn.Parameter(torch.zeros(24, self.num_prompt_tokens, prompt_dim))  # 24 layers
        # Xavier initialization for prompt_embedding
        val = math.sqrt(6. / float(2 * prompt_dim))  # Xavier initialization factor
        nn.init.uniform_(self.prompt_embedding.data, -val, val)
        # Dropout layer for the prompt
        self.prompt_dropout = nn.Dropout(p=dropout)
        self.visual = visual
        
    def forward(self, audio_data):
        # Process the input audio using Wav2Vec2 Feature Extractor
        feat = self.processor(audio_data, sampling_rate=self.sampling_rate, return_tensors="pt").input_values.to(self.device)
        feat = feat.squeeze(dim=0)  

        with torch.no_grad():
            feat = self.model.feature_extractor(feat)
            feat = feat.transpose(1, 2)
            # Feature projection
            hidden_state, extract_features = self.model.feature_projection(feat)
            if self.visual:
                first_hidden_state = hidden_state
            position_embeddings = self.model.encoder.pos_conv_embed(hidden_state)
            hidden_state = hidden_state + position_embeddings
            hidden_state = self.model.encoder.dropout(hidden_state)

        if self.visual:
            all_self_attentions = []
        B = feat.size(0)  
        for i in range(self.model.config.num_hidden_layers):
            if i == 0:
                prompt = self.prompt_embedding[i].expand(B, -1, -1).to(self.device)
                prompt = self.prompt_dropout(prompt) 
                hidden_state = torch.cat((prompt, hidden_state), dim=1)
                if self.visual:
                    print(hidden_state.shape, 'hidden_state')
                    hidden_state, attention_weight = self.model.encoder.layers[i](hidden_state, output_attentions=self.visual)
                    all_self_attentions.append(attention_weight)
                else:
                    hidden_state = self.model.encoder.layers[i](hidden_state)[0]
            else:    
                prompt = self.prompt_embedding[i].expand(B, -1, -1).to(self.device)
                prompt = self.prompt_dropout(prompt)  
                hidden_state = torch.cat((prompt, hidden_state[:, self.num_prompt_tokens:, :]), dim=1)
                if self.visual: 
                    hidden_state, attention_weight = self.model.encoder.layers[i](hidden_state, output_attentions=self.visual)
                    all_self_attentions.append(attention_weight)  
                else:
                    hidden_state = self.model.encoder.layers[i](hidden_state)[0]  

        if self.visual:
            print(len(all_self_attentions), "all_self_attentions")
            return first_hidden_state, hidden_state,all_self_attentions
        else:
            return hidden_state

    def extract_features(self, audio_data):
        # Process the input audio and extract the features using the forward pass
        return self.forward(audio_data)  # Return the final layer's output

    
class PT_WAVLM(nn.Module):
    def __init__(self, model_dir, prompt_dim, device='cuda', sampling_rate=16000, num_prompt_tokens=10, dropout=0.1, visual=False):
        super(PT_WAVLM, self).__init__()

        # Set device (GPU or CPU)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sampling_rate = sampling_rate

        # Load the pre-trained model configuration and weights
        self.config = WavLMConfig.from_json_file(f"{model_dir}/config.json")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir, do_normalize = False)
        self.model = WavLMModel.from_pretrained(model_dir).to(self.device)

        # Enable output of hidden states
        self.model.config.output_hidden_states = True
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Create a learnable prompt embedding for 24 layers
        self.prompt_dim = prompt_dim
        self.num_prompt_tokens = num_prompt_tokens  # Assume prompt consists of 10 tokens
        self.prompt_embedding = nn.Parameter(torch.zeros(24, self.num_prompt_tokens, prompt_dim))  # 24 layers
        # Xavier initialization for prompt_embedding
        val = math.sqrt(6. / float(2 * prompt_dim))  # Xavier initialization factor
        nn.init.uniform_(self.prompt_embedding.data, -val, val)
        # Dropout layer for the prompt
        self.prompt_dropout = nn.Dropout(p=dropout)
        self.visual = visual

    def forward(self, audio_data):
        # Process the input audio using Wav2Vec2 Feature Extractor
        feat = self.processor(audio_data, sampling_rate=self.sampling_rate, return_tensors="pt").input_values.to(self.device)
        feat = feat.squeeze(dim=0)  

        with torch.no_grad():
            outputs = self.model(feat)
            hidden_states = outputs.hidden_states  # A tuple of hidden states from all layers
        hidden_state = hidden_states[0]
          
        B = feat.size(0)  
        for i in range(self.model.config.num_hidden_layers):
            if i == 0:
                prompt = self.prompt_embedding[i].expand(B, -1, -1).to(self.device)
                prompt = self.prompt_dropout(prompt) 
                hidden_state = torch.cat((prompt, hidden_state), dim=1)
                hidden_state = self.model.encoder.layers[i](hidden_state)[0]
            else:    
                prompt = self.prompt_embedding[i].expand(B, -1, -1).to(self.device)
                prompt = self.prompt_dropout(prompt)  
                hidden_state = torch.cat((prompt, hidden_state[:, self.num_prompt_tokens:, :]), dim=1)
                hidden_state = self.model.encoder.layers[i](hidden_state)[0]

        if self.visual:
            encoder_outputs = self.model.encoder(
            hidden_states=hidden_state,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )
            attention_weights = encoder_outputs.attentions
            return hidden_state,attention_weights
        else:
            return hidden_state

    def extract_features(self, audio_data):
        # Process the input audio and extract the features using the forward pass
        return self.forward(audio_data)  # Return the final layer's output


class PT_MERT(nn.Module):
    def __init__(self, model_dir, prompt_dim, device='cuda', sampling_rate=16000, num_prompt_tokens=10, dropout=0.1, visual=False):
        super(PT_MERT, self).__init__()

        # Set device (GPU or CPU)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sampling_rate = sampling_rate

        # Load the pre-trained model configuration and weights
        self.config = Wav2Vec2Config.from_json_file(f"{model_dir}/config.json")
        self.processor = AutoFeatureExtractor.from_pretrained(model_dir, sampling_rate = 16000,do_normalize = False)
        self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(self.device)

        # Enable output of hidden states
        self.model.config.output_hidden_states = True
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Create a learnable prompt embedding for 24 layers
        self.prompt_dim = prompt_dim
        self.num_prompt_tokens = num_prompt_tokens  # Assume prompt consists of 10 tokens
        self.prompt_embedding = nn.Parameter(torch.zeros(24, self.num_prompt_tokens, prompt_dim))  # 24 layers

        # Xavier initialization for prompt_embedding
        val = math.sqrt(6. / float(2 * prompt_dim))  # Xavier initialization factor
        nn.init.uniform_(self.prompt_embedding.data, -val, val)

        # Dropout layer for the prompt
        self.prompt_dropout = nn.Dropout(p=dropout)
        self.visual = visual

    def forward(self, audio_data):
        # Process the input audio using Wav2Vec2 Feature Extractor
        feat = self.processor(audio_data, sampling_rate=self.sampling_rate, return_tensors="pt").input_values.to(self.device)
        feat = feat.squeeze(dim=0)  

        with torch.no_grad():
            outputs = self.model(feat)
            hidden_states = outputs.hidden_states  # A tuple of hidden states from all layers
        hidden_state = hidden_states[0]
          
        B = feat.size(0)  
        for i in range(self.model.config.num_hidden_layers):
            if i == 0:
                prompt = self.prompt_embedding[i].expand(B, -1, -1).to(self.device)
                prompt = self.prompt_dropout(prompt) 
                hidden_state = torch.cat((prompt, hidden_state), dim=1)
                hidden_state = self.model.encoder.layers[i](hidden_state)[0]
            else:    
                prompt = self.prompt_embedding[i].expand(B, -1, -1).to(self.device)
                prompt = self.prompt_dropout(prompt)  
                hidden_state = torch.cat((prompt, hidden_state[:, self.num_prompt_tokens:, :]), dim=1)
                hidden_state = self.model.encoder.layers[i](hidden_state)[0]

        if self.visual:
            encoder_outputs = self.model.encoder(
            hidden_states=hidden_state,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )
            attention_weights = encoder_outputs.attentions
            return hidden_state,attention_weights
        else:
            return hidden_state

    def extract_features(self, audio_data):
        # Process the input audio and extract the features using the forward pass
        return self.forward(audio_data)  # Return the final layer's output


class BEATs(nn.Module):
    def __init__(
        self,
        model_dir,
        device='cuda',
        sampling_rate=16000,
        freeze=True,
        return_hidden_states=False,
        selected_layers=None,
        layer_fusion="last",
    ):
        super(BEATs, self).__init__()

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sampling_rate = sampling_rate
        self.freeze = freeze
        self.return_hidden_states = return_hidden_states
        self.selected_layers = (
            tuple(int(i) for i in selected_layers) if selected_layers is not None else None
        )

        self.model = BEATsModel(
            cfg_path=f"{model_dir}/BEATs_iter3_plus_AS2M.pt"
        ).to(self.device)   # move to device at construction time

        self.hidden_size = self.model.model.cfg.encoder_embed_dim
        self.num_hidden_states = self.model.model.cfg.encoder_layers + 1

        lf = str(layer_fusion).strip().lower()
        if lf == "cat":
            lf = "cat_proj_v2"
        elif lf == "cat_proj":
            lf = "cat_proj_v2"
        self.layer_fusion = lf
        if lf in (
            "cat_linear",
            "cat_proj_v1",
            "cat_proj_v2",
            "mean",
            "weight_sum",
        ):
            if not self.selected_layers:
                raise ValueError(
                    "BEATs: selected_layers must be a non-empty sequence when "
                    f"layer_fusion={lf!r}"
                )
        elif lf == "last" and self.selected_layers is not None:
            if len(self.selected_layers) > 1:
                raise ValueError(
                    "BEATs: layer_fusion='last' accepts at most one "
                    f"selected_layers index; got {self.selected_layers!r}"
                )
        elif lf != "last":
            raise ValueError(
                f"Unsupported BEATs layer_fusion={lf!r}. Choose from "
                "['last', 'cat_linear', 'cat_proj_v1', 'cat_proj_v2', "
                "'mean', 'weight_sum']."
            )

        n_sel = len(self.selected_layers) if self.selected_layers else 0
        self.per_layer_ln = None
        self.layer_proj = None
        self.cat_linear_head = None
        self.layer_weights = None

        if lf == "cat_linear" and n_sel > 0:
            self.cat_linear_head = nn.Linear(
                self.hidden_size * n_sel, self.hidden_size
            )
        elif lf == "cat_proj_v1" and n_sel > 0:
            self.layer_proj = nn.Sequential(
                nn.LayerNorm(self.hidden_size * n_sel),
                nn.Linear(self.hidden_size * n_sel, self.hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
            )
        elif lf == "cat_proj_v2" and n_sel > 0:
            self.per_layer_ln = nn.ModuleList(
                [nn.LayerNorm(self.hidden_size) for _ in range(n_sel)]
            )
            self.layer_proj = nn.Sequential(
                nn.LayerNorm(self.hidden_size * n_sel),
                nn.Linear(self.hidden_size * n_sel, self.hidden_size),
                nn.Dropout(0.1),
            )
        elif lf == "weight_sum" and n_sel > 0:
            self.layer_weights = nn.Parameter(torch.zeros(n_sel))

        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.model.train()

    def _fuse_hidden_states(self, last_hidden_state, hidden_states):
        n_h = len(hidden_states)

        if self.layer_fusion == "last":
            if not self.selected_layers:
                return last_hidden_state, hidden_states
            idx = self.selected_layers[0]
            if idx < 0 or idx >= n_h:
                raise IndexError(
                    f"BEATs selected_layers index {idx} out of range for "
                    f"{n_h} hidden_states (valid 0..{n_h - 1})."
                )
            return hidden_states[idx], hidden_states

        for i in self.selected_layers:
            if i < 0 or i >= n_h:
                raise IndexError(
                    f"BEATs selected_layers index {i} out of range for "
                    f"{n_h} hidden_states (valid 0..{n_h - 1})."
                )

        selected = [hidden_states[i] for i in self.selected_layers]

        if self.layer_fusion == "cat_linear":
            fused = torch.cat(selected, dim=-1)
            fused = self.cat_linear_head(fused)
        elif self.layer_fusion == "cat_proj_v1":
            fused = torch.cat(selected, dim=-1)
            fused = self.layer_proj(fused)
        elif self.layer_fusion == "cat_proj_v2":
            normed = [
                ln(h) for ln, h in zip(self.per_layer_ln, selected)
            ]
            fused = torch.cat(normed, dim=-1)
            fused = self.layer_proj(fused)
        elif self.layer_fusion == "mean":
            fused = torch.stack(selected, dim=0).mean(dim=0)
        elif self.layer_fusion == "weight_sum":
            weights = F.softmax(self.layer_weights, dim=0)
            fused = sum(w * h for w, h in zip(weights, selected))
        else:
            raise ValueError(f"Unsupported BEATs layer_fusion={self.layer_fusion!r}")

        return fused, hidden_states

    def extract_features(self, input_data):
        # input should be in shape (batch, length)
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data

        need_hidden = self.return_hidden_states or self.selected_layers is not None
        if need_hidden:
            # BEATs hidden-state index convention matches XLSR here:
            # 0 = encoder input, 1..N = outputs after transformer layers.
            max_idx = (
                max(self.selected_layers)
                if self.selected_layers is not None
                else self.num_hidden_states - 1
            )
            emb, hidden_states = self.model(
                input_tmp,
                tgt_layer=max_idx,
                return_hidden_states=True,
            )
            emb, hidden_states = self._fuse_hidden_states(emb, hidden_states)
            if self.return_hidden_states:
                return emb, hidden_states
            return emb

        # [batch, length, dim]
        return self.model(input_tmp)


class CLAP(nn.Module):
    def __init__(self, model_dir, device='cuda', freeze=True, sampling_rate=16000, return_hidden_states=False):
        super(CLAP, self).__init__()
        
        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu'
        )
        self.sampling_rate = sampling_rate
        self.return_hidden_states = return_hidden_states
    
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=sampling_rate,
            new_freq=48000
        ).to(self.device)
        
        self.processor = ClapProcessor.from_pretrained(model_dir)
        self.model = ClapModel.from_pretrained(model_dir).to(self.device)
            
        # only use audio encoder
        self.audio_encoder = self.model.audio_model# HTSAT
        
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.freeze = freeze
    
    def forward(self, audio_data):
        """
        Args:
            audio_data: [B, T]
        Returns:[B, C=1024, 2, 32]
        """
        if isinstance(audio_data, np.ndarray):
            audio_data = torch.tensor(audio_data, dtype=torch.float32)
        
        if audio_data.dim() == 1:
            audio_data = audio_data.unsqueeze(0)# [1, T]
        
        audio_data = audio_data.to(self.device)
        if audio_data.shape[1] != 48000:
            audio_data = self.resampler(audio_data)
            
        # CLAP's processor requires numpy input
        audio_np = audio_data.cpu().numpy()
        
        inputs = self.processor(
            audios=audio_np,
            sampling_rate=48000,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        if self.freeze:
            with torch.no_grad():
                outputs = self.audio_encoder(**inputs,
                output_hidden_states=self.return_hidden_states
                )
        else:
            outputs = self.audio_encoder(
                **inputs,
                output_hidden_states=self.return_hidden_states
            )

        last_hidden = outputs.last_hidden_state  # [B, C, F, T]
        clap_feat = last_hidden.mean(dim=2).permute(0, 2, 1)  # [B, T, C]

        if self.return_hidden_states:
            return clap_feat, outputs.hidden_states
        return clap_feat

    def extract_features(self, audio_data):
        return self.forward(audio_data)


def _clap_preprocess(feat: torch.Tensor) -> torch.Tensor:
    """
    No-op pass-through for CLAP features.

    CLAP.forward already returns ``(B, T, C)``-shaped features, so no
    additional pre-processing is needed before the AASIST backend.
    Kept as a hook in case future CLAP variants need reshaping.
    """
    return feat
