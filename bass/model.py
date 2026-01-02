"""
BASS: Block-wise Adaptation for Speech Summarization
=====================================================

Implementation based on:
"BASS: Block-wise Adaptation for Speech Summarization"
Sharma et al., 2023

Key Components:
1. BlockProcessor: Splits long audio into blocks
2. SemanticUpdater: Passes semantic context across blocks
3. BlockwiseEncoder: Encodes each block (uses pretrained ESPNet ASR encoder)
4. BlockwiseDecoder: Generates summary from each block
5. BASSModel: Main model that processes blocks incrementally
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math

@dataclass
class BASSConfig:
    """Configuration for BASS model"""
    # Audio/Encoder configuration
    freeze_encoder: bool = True
    input_dim: int = 40  # Mel spectrogram features dimension
    encoder_dim: int = 2048  # Output dimension of pretrained ASR encoder
    
    # Block configuration
    block_size: int = 1000  # Number of frames per block
    block_overlap: int = 0  # Overlap between blocks (0 for no overlap)
    
    # Semantic context configuration
    semantic_dim: int = 2048  # Dimension of semantic representation
    num_semantic_layers: int = 2  # Number of layers in semantic updater
    
    # Decoder configuration
    vocab_size: int = 500  # Vocabulary size for text generation
    decoder_dim: int = 2048  # Decoder hidden dimension
    decoder_num_layers: int = 6  # Number of decoder layers
    decoder_num_heads: int = 8  # Number of attention heads
    decoder_ffn_dim: int = 2048  # Feed-forward dimension
    
    # Training configuration
    max_summary_length: int = 512  # Maximum summary length
    label_smoothing: float = 0.15
    beam_size: int = 8
    model_averaging: bool = False

    dropout: float = 0.1
    use_layer_norm: bool = True
    
    # Pretrained encoder path (optional)
    pretrained_encoder_path: Optional[str] = None
    
    # Ablation flags (for experiments)
    use_context_carry: bool = True  # If False, reset semantic context for each block (block independent)
    use_stop_gradient: bool = False  # If True, detach previous semantic context before use


class SemanticUpdater(nn.Module):
    """
    Semantic Updater: Aggregates and updates semantic context across blocks
    
    The semantic representation captures the meaning from acoustic features
    and is not affected by how it's expressed. This allows the model to
    integrate information from previous blocks into the current block processing.
    """
    
    def __init__(self, 
                 encoder_dim: int,
                 semantic_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.encoder_dim = encoder_dim
        self.semantic_dim = semantic_dim
        
        # Project encoder output to semantic space
        self.encoder_to_semantic = nn.Linear(encoder_dim, semantic_dim)

        # Lightweight aggregation: Simple MLP instead of heavy transformer
        # BASS paper uses simple pooling + projection, not full transformer
        aggregation_layers = []
        for i in range(num_layers):
            aggregation_layers.extend([
                nn.Linear(semantic_dim, semantic_dim),
                nn.LayerNorm(semantic_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        self.aggregation_mlp = nn.Sequential(*aggregation_layers)

        # Layer norm
        self.layer_norm = nn.LayerNorm(semantic_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.encoder_to_semantic.weight)
        nn.init.zeros_(self.encoder_to_semantic.bias)
    
    def forward(self,
                current_block_encoding: torch.Tensor,
                previous_semantic_context: Optional[torch.Tensor] = None,
                use_stop_gradient: bool = False,) -> torch.Tensor:
        """
        Update semantic context with current block encoding
        
        Args:
            current_block_encoding: Encoder output for current block
                Shape: [batch, block_seq_len, encoder_dim]
            previous_semantic_context: Semantic context from previous block(s)
                Shape: [batch, num_prev_latents, semantic_dim] or None
            use_stop_gradient: if True, detach previous_semantic_context during training
                
        Returns:
            updated_semantic_context: Updated semantic representation
                Shape: [batch, num_latents, semantic_dim]
                Note: num_latents may vary if context grows, but typically we keep it fixed
        """
        batch_size = current_block_encoding.size(0)
        
        # Project encoder output to semantic space
        current_semantic = self.encoder_to_semantic(current_block_encoding)
        # Shape: [batch, block_seq_len, semantic_dim]

        # Mean pool first (lightweight)
        pooled_current = current_semantic.mean(dim=1, keepdim=True)
        # Shape: [batch, 1, semantic_dim]

        # Apply MLP for aggregation (operates on pooled vector)
        aggregated_current = self.aggregation_mlp(pooled_current)
        # Shape: [batch, 1, semantic_dim]
        
        # Combine with previous context if available
        if previous_semantic_context is not None:
            # Concatenate previous context with current aggregated representation
            # 이전 컨텍스트는 상태로만 사용 (훈련 시, 옵션)
            if self.training and use_stop_gradient:
                previous_semantic_context = previous_semantic_context.detach()
            combined_context = torch.cat([previous_semantic_context, aggregated_current], dim=1)

            # Shape: [batch, num_prev_latents + 1, semantic_dim]
            
            # For efficiency, optionally limit context size (e.g., keep only last N)
            # Here we keep all context but this can be made configurable
            max_context_size = 128  # Limit to prevent unbounded growth
            if combined_context.size(1) > max_context_size:
                # Keep the most recent context
                combined_context = combined_context[:, -max_context_size:, :]
            
            output_context = combined_context
        else:
            # First block: just use current aggregated representation
            output_context = aggregated_current
        
        # Apply layer norm
        output_context = self.layer_norm(output_context)
        
        return output_context


class BlockwiseEncoder(nn.Module):
    """
    Blockwise Encoder: Wraps pretrained ESPNet ASR encoder
    
    This encoder processes each block independently and extracts
    acoustic features that will be converted to semantic representations.
    """
    
    def __init__(self,
                 encoder: nn.Module,
                 input_dim: int = 40,
                 output_dim: int = 512):
        """
        Args:
            encoder: Pretrained ESPNet ASR encoder
            input_dim: Input feature dimension
            output_dim: Output encoding dimension
        """
        super().__init__()
        
        self.encoder = encoder
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Freeze encoder if it's pretrained
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(self,
                audio_features: torch.Tensor,
                audio_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode audio features
        
        Args:
            audio_features: Audio features [batch, seq_len, input_dim]
            audio_lengths: Actual lengths [batch]
            
        Returns:
            encoded_features: Encoded features [batch, encoded_seq_len, output_dim]
            encoded_lengths: Encoded lengths [batch]
        """
        # Forward through pretrained encoder
        # ESPNet encoders typically return (output, lengths, ...)
        encoder_output = self.encoder(audio_features, audio_lengths)
        
        if isinstance(encoder_output, tuple):
            encoded_features = encoder_output[0]
            encoded_lengths = encoder_output[1] if len(encoder_output) > 1 else audio_lengths
        else:
            encoded_features = encoder_output
            encoded_lengths = audio_lengths
        
        return encoded_features, encoded_lengths


class BlockwiseDecoder(nn.Module):
    """
    Blockwise Decoder: Generates summary text from semantic context
    
    The decoder uses the semantic representation from the current block
    (which includes information from previous blocks) to generate a summary.
    Each block produces a complete summary, which can be fully modified
    based on new information in the current block.
    """
    
    def __init__(self,
                 semantic_dim: int,
                 vocab_size: int,
                 decoder_dim: int = 512,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 ffn_dim: int = 2048,
                 max_length: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        self.semantic_dim = semantic_dim
        self.vocab_size = vocab_size
        self.decoder_dim = decoder_dim
        self.max_length = max_length
        
        # Embedding layer
        self.token_embedding = nn.Embedding(vocab_size, decoder_dim)
        self.positional_embedding = nn.Embedding(max_length, decoder_dim)
        
        # Project semantic context to decoder dimension
        self.semantic_projection = nn.Linear(semantic_dim, decoder_dim)
        
        # Transformer decoder layers with cross-attention to semantic context
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True
        )
        self.decoder_layers = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(decoder_dim, vocab_size)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(decoder_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.positional_embedding.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.semantic_projection.weight)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.semantic_projection.bias)
        nn.init.zeros_(self.output_projection.bias)
    
    def forward(self,
            semantic_context: torch.Tensor,
            target_ids: Optional[torch.Tensor] = None,
            max_length: Optional[int] = None,
            num_beams: int = 1,
            length_penalty: float = 1.0) -> torch.Tensor:
        """
        Generate summary from semantic context
        
        Args:
            semantic_context: Semantic representation
                Shape: [batch, num_latents, semantic_dim]
            target_ids: Target token ids for training [batch, target_len]
            max_length: Maximum generation length (default: self.max_length)
            
        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
        """
        batch_size = semantic_context.size(0)
        max_len = max_length or self.max_length
        
        # Project semantic context to decoder dimension
        memory = self.semantic_projection(semantic_context)  # [batch, num_latents, decoder_dim]
        memory = self.layer_norm(memory)
        
        if target_ids is not None:
            # Training: use teacher forcing
            seq_len = target_ids.size(1)
            
            # Embed tokens
            token_embeds = self.token_embedding(target_ids)  # [batch, seq_len, decoder_dim]
            
            # Add positional embeddings
            positions = torch.arange(seq_len, device=target_ids.device).unsqueeze(0).expand(batch_size, -1)
            pos_embeds = self.positional_embedding(positions)
            decoder_input = token_embeds + pos_embeds
            
            # Create causal mask for decoder
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(target_ids.device)
            
            # Forward through decoder
            decoder_output = self.decoder_layers(
                tgt=decoder_input,
                memory=memory,
                tgt_mask=tgt_mask
            )
            
            # Project to vocab
            logits = self.output_projection(decoder_output)
            
            return logits
        else:
            # Inference: autoregressive generation
            sos_token_id = 0
            eos_token_id = 1

            max_len = max_length or self.max_length
            device = semantic_context.device

            # Helper: run one forward step to get next-token logprobs for a given prefix
            def _next_logprobs(prefix_ids: torch.Tensor) -> torch.Tensor:
                # prefix_ids: [1, cur_len]
                cur_len = prefix_ids.size(1)

                token_embeds = self.token_embedding(prefix_ids)  # [1, cur_len, d]
                positions = torch.arange(cur_len, device=device).unsqueeze(0)
                pos_embeds = self.positional_embedding(positions)
                decoder_input = token_embeds + pos_embeds

                tgt_mask = nn.Transformer.generate_square_subsequent_mask(cur_len).to(device)

                dec_out = self.decoder_layers(
                    tgt=decoder_input,
                    memory=memory[:1],          # memory for this sample, set by outer scope
                    tgt_mask=tgt_mask
                )
                logits = self.output_projection(dec_out[:, -1, :])  # [1, vocab]
                return F.log_softmax(logits, dim=-1)                # [1, vocab]

            # Beam search per sample (간단/안전 버전: batch를 샘플 단위로 루프)
            generated = []
            for b in range(batch_size):
                # memory를 샘플별로 바꿔치기 해서 _next_logprobs가 쓰게 함
                memory = self.semantic_projection(semantic_context[b:b+1])
                memory = self.layer_norm(memory)

                beams = [(torch.tensor([[sos_token_id]], device=device, dtype=torch.long), 0.0, False)]
                for step in range(max_len - 1):
                    candidates = []
                    all_finished = True

                    for seq, score, finished in beams:
                        if finished:
                            candidates.append((seq, score, True))
                            continue

                        all_finished = False
                        logp = _next_logprobs(seq)  # [1, vocab]
                        topk_logp, topk_ids = torch.topk(logp, k=num_beams, dim=-1)  # [1, K]

                        for k in range(num_beams):
                            next_id = topk_ids[0, k].view(1, 1)
                            next_seq = torch.cat([seq, next_id], dim=1)
                            next_score = score + float(topk_logp[0, k].item())
                            next_finished = (int(next_id.item()) == eos_token_id)
                            candidates.append((next_seq, next_score, next_finished))

                    # if all beams finished, stop early
                    if all_finished:
                        break

                    # select top beams by length-penalized score
                    def _rank(item):
                        seq, s, fin = item
                        lp = (seq.size(1) ** length_penalty) if length_penalty != 1.0 else 1.0
                        return s / lp

                    candidates.sort(key=_rank, reverse=True)
                    beams = candidates[:num_beams]

                best_seq, best_score, _ = max(
                    beams,
                    key=lambda x: (x[1] / ((x[0].size(1) ** length_penalty) if length_penalty != 1.0 else 1.0))
                )
                generated.append(best_seq)

            # pad to same length across batch
            max_out_len = max(seq.size(1) for seq in generated)
            out = torch.full((batch_size, max_out_len), eos_token_id, device=device, dtype=torch.long)
            for b, seq in enumerate(generated):
                out[b, :seq.size(1)] = seq[0]
            return out

class BlockProcessor:
    """
    Block Processor: Splits long audio sequences into blocks
    
    This handles the chunking of long audio inputs into fixed-size blocks
    that can be processed sequentially.
    """
    
    def __init__(self,
                 block_size: int,
                 overlap: int = 0):
        """
        Args:
            block_size: Number of frames per block
            overlap: Number of overlapping frames between consecutive blocks
        """
        self.block_size = block_size
        self.overlap = overlap
    
    def split_into_blocks(self,
                          audio_features: torch.Tensor,
                          audio_lengths: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Split audio into blocks
        
        Args:
            audio_features: Audio features [batch, max_seq_len, feat_dim]
            audio_lengths: Actual lengths [batch]
            
        Returns:
            blocks: List of (block_features, block_lengths) tuples
                Each block_features: [batch, block_seq_len, feat_dim]
                Each block_lengths: [batch]
        """
        batch_size, max_seq_len, feat_dim = audio_features.shape
        blocks = []
        
        for batch_idx in range(batch_size):
            actual_len = audio_lengths[batch_idx].item()
            batch_blocks = []
            
            start = 0
            while start < actual_len:
                end = min(start + self.block_size, actual_len)
                block_features = audio_features[batch_idx:batch_idx+1, start:end, :]
                block_length = torch.tensor([end - start], device=audio_features.device)
                batch_blocks.append((block_features, block_length))
                
                start = end - self.overlap  # Move forward, accounting for overlap
            
            blocks.append(batch_blocks)
        
        # Convert to format where each element contains all batches for that block index
        num_blocks_per_batch = max(len(batch_blocks) for batch_blocks in blocks)
        block_list = []
        
        for block_idx in range(num_blocks_per_batch):
            block_features_list = []
            block_lengths_list = []
            
            # Find max block length for this block index across all batches
            max_block_len = 0
            for batch_idx in range(batch_size):
                if block_idx < len(blocks[batch_idx]):
                    feat, length = blocks[batch_idx][block_idx]
                    if length.item() > max_block_len:
                        max_block_len = length.item()
            
            # Now collect all blocks, padding to max_block_len
            for batch_idx in range(batch_size):
                if block_idx < len(blocks[batch_idx]):
                    feat, length = blocks[batch_idx][block_idx]
                    block_len = length.item()
                    
                    # Pad if necessary to match max_block_len
                    if block_len < max_block_len:
                        padding_size = max_block_len - block_len
                        pad_tensor = torch.zeros(1, padding_size, feat_dim, device=feat.device)
                        feat = torch.cat([feat, pad_tensor], dim=1)
                        length = torch.tensor([max_block_len], device=length.device, dtype=torch.long)
                    
                    block_features_list.append(feat)
                    block_lengths_list.append(length)
                else:
                    # Padding for batches with fewer blocks
                    empty_feat = torch.zeros(1, max_block_len if max_block_len > 0 else 1, feat_dim, device=audio_features.device)
                    empty_length = torch.tensor([0], device=audio_features.device, dtype=torch.long)
                    block_features_list.append(empty_feat)
                    block_lengths_list.append(empty_length)
            
            # Concatenate across batch dimension
            block_features = torch.cat(block_features_list, dim=0)
            block_lengths = torch.cat(block_lengths_list, dim=0)
            block_list.append((block_features, block_lengths))
        
        return block_list


class BASSModel(nn.Module):
    """
    BASS: Block-wise Adaptation for Speech Summarization
    
    Main model that processes long audio sequences in blocks, generating
    summaries after each block and updating semantic context across blocks.
    """
    
    def __init__(self,
                 config: BASSConfig,
                 pretrained_encoder: Optional[nn.Module] = None):
        """
        Args:
            config: BASS configuration
            pretrained_encoder: Pretrained ESPNet ASR encoder (optional)
        """
        super().__init__()
        
        self.config = config
        
        # Block processor
        self.block_processor = BlockProcessor(
            block_size=config.block_size,
            overlap=config.block_overlap
        )
        
        # Blockwise encoder (wraps pretrained ASR encoder)
        if pretrained_encoder is not None:
            self.encoder = BlockwiseEncoder(
                encoder=pretrained_encoder,
                input_dim=config.input_dim,
                output_dim=config.encoder_dim
            )
        else:
            # Create a simple placeholder encoder if none provided
            # In practice, you should provide a pretrained encoder
            self.encoder = None
            raise ValueError("pretrained_encoder must be provided")
        
        if config.freeze_encoder: # freeze encoder if True
            for p in self.encoder.parameters():
                p.requires_grad = False
        
        # Semantic updater
        self.semantic_updater = SemanticUpdater(
            encoder_dim=config.encoder_dim,
            semantic_dim=config.semantic_dim,
            num_layers=config.num_semantic_layers,
            dropout=config.dropout
        )
        
        # Blockwise decoder
        self.decoder = BlockwiseDecoder(
            semantic_dim=config.semantic_dim,
            vocab_size=config.vocab_size,
            decoder_dim=config.decoder_dim,
            num_layers=config.decoder_num_layers,
            num_heads=config.decoder_num_heads,
            ffn_dim=config.decoder_ffn_dim,
            max_length=config.max_summary_length,
            dropout=config.dropout
        )
    
    def forward(self,
                audio_features: torch.Tensor,
                audio_lengths: torch.Tensor,
                target_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                backward_per_block: bool = False
        ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through BASS model

        Args:
            audio_features: Audio features [batch, max_seq_len, feat_dim]
            audio_lengths: Actual audio lengths [batch]
            target_ids: Target summary token ids for training [batch, target_len]
            labels: Labels for loss computation (with padding masked as -100) [batch, target_len]
            backward_per_block: If True, call backward() per block to save memory

        Returns:
            Dictionary containing:
            - logits: Output logits from final block [batch, seq_len, vocab_size]
            - block_logits: List of logits from each block (for training)
            - semantic_contexts: List of semantic contexts from each block
            - loss: Average loss across blocks (if labels provided)
            - loss_value: Scalar loss value for logging (if labels provided)
            - num_blocks: Number of blocks processed
        """
        # Split audio into blocks
        blocks = self.block_processor.split_into_blocks(audio_features, audio_lengths)
        
        # Process blocks sequentially
        previous_semantic_context = None
        block_logits = []
        semantic_contexts = []
        
        for block_idx, (block_features, block_lengths) in enumerate(blocks):
            # Skip empty blocks or blocks too short for encoder subsampling
            if block_lengths.sum() == 0 or block_lengths.max() < 8:
                continue

            # Encode current block
            if self.config.freeze_encoder:
                with torch.no_grad():
                    block_encoding, block_encoding_lengths = self.encoder(block_features, block_lengths)
            else:
                block_encoding, block_encoding_lengths = self.encoder(block_features, block_lengths)

            # Update semantic context: concat(prev, cur) 유지
            # Note: previous_semantic_context는 이전 반복에서 설정됨
            # use_context_carry=False면 이전 반복에서 None으로 설정되어 있음
            current_semantic_context = self.semantic_updater(
                current_block_encoding=block_encoding,
                previous_semantic_context=previous_semantic_context,
                use_stop_gradient=getattr(self.config, "use_stop_gradient", False),
            )

            # === 핵심: "previous block embedding만" 다음 블록에 전달 ===
            # current_semantic_context: [B, L, D] (보통 L=1 또는 2)
            prev_only = current_semantic_context[:, -1:, :]  # [B, 1, D]

            # 디버깅/분석용으로 무엇을 저장할지 선택
            # (논문 문장 그대로 가려면 prev_only 저장이 더 깔끔)
            # Only save contexts when not using per-block backward (to save memory)
            if not backward_per_block:
                semantic_contexts.append(prev_only)

            # 다음 블록에 들어갈 previous_semantic_context 설정
            # carry ON일 때만 다음 블록으로 전달
            if getattr(self.config, "use_context_carry", True):
                # IMPORTANT: Always detach when using backward_per_block to avoid OOM
                # Per-block backward requires breaking gradient flow between blocks
                if self.training and (getattr(self.config, "use_stop_gradient", False) or backward_per_block):
                    previous_semantic_context = prev_only.detach()
                else:
                    previous_semantic_context = prev_only
            else:
                previous_semantic_context = None

            # Generate summary from current semantic context
            # (decoder는 "현재 블록까지 반영된 컨텍스트"를 쓰는 게 맞으니 current_semantic_context 사용)
            if target_ids is not None:
                # Training/validation with teacher forcing
                block_logit = self.decoder(
                    semantic_context=current_semantic_context,
                    target_ids=target_ids
                )

                # Compute loss if labels provided
                if labels is not None:
                    # Shift for next-token prediction (ESPnet-compatible)
                    # Decoder sees [BOS, tok1, ..., tokN] and predicts [tok1, tok2, ..., tokN, EOS]
                    shift_logits = block_logit[:, :-1, :].contiguous()  # Remove last position
                    shift_labels = labels[:, 1:].contiguous()            # Remove BOS

                    # Compute cross-entropy loss
                    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=self.config.label_smoothing)
                    block_loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )

                    # Per-block backward to save memory
                    if backward_per_block:
                        # Backward immediately to free computation graph
                        # No retain_graph needed - each block is independent (context is detached)
                        # PyTorch automatically accumulates gradients across multiple backward() calls
                        (block_loss / len(blocks)).backward()

                        # Accumulate scalar loss value only
                        if block_idx == 0:
                            total_loss_value = block_loss.item()
                        else:
                            total_loss_value += block_loss.item()

                        # Detach logits to free computation graph
                        final_logits = block_logit.detach()

                        # Clear intermediate tensors to free memory
                        del block_loss, shift_logits, shift_labels, block_logit

                        # Also clear encoder outputs if not needed
                        if block_idx < len(blocks) - 1:  # Not last block
                            del block_encoding, current_semantic_context

                        # Force GPU cache cleanup periodically
                        if torch.cuda.is_available() and (block_idx + 1) % 10 == 0:
                            torch.cuda.empty_cache()

                        # Store detached logits for return
                        block_logits.append(final_logits)
                    else:
                        # Standard: accumulate loss and logits
                        if block_idx == 0:
                            total_loss = block_loss
                        else:
                            total_loss += block_loss
                        block_logits.append(block_logit)
                else:
                    # No labels - just store logits
                    block_logits.append(block_logit)
            else:
                # Inference mode - generate with beam search
                with torch.no_grad():
                    generated_ids = self.decoder(
                        semantic_context=current_semantic_context,
                        target_ids=None,
                        num_beams=getattr(self.config, "beam_size", 1)
                    )
                block_logits.append(generated_ids)

        # Return final block output (and optionally all blocks for training)
        result = {
            "logits": block_logits[-1] if block_logits else None,
            "block_logits": block_logits,
            "semantic_contexts": semantic_contexts,
            "num_blocks": len([b for b in blocks if b[1].sum() > 0 and b[1].max() >= 8])  # Count non-empty blocks
        }

        # Add loss if computed
        if labels is not None:
            if backward_per_block:
                # Return average loss across blocks (scalar value for logging)
                result["loss_value"] = total_loss_value / len(blocks) if 'total_loss_value' in locals() else 0.0
                result["loss"] = result["loss_value"]  # Also set loss for consistency
            else:
                # Return accumulated loss (tensor)
                avg_loss = total_loss / len(blocks) if 'total_loss' in locals() else torch.tensor(0.0)
                result["loss"] = avg_loss
                result["loss_value"] = avg_loss.item()

        return result
    
    def generate(self,
                 audio_features: torch.Tensor,
                 audio_lengths: torch.Tensor,
                 max_length: Optional[int] = None) -> torch.Tensor:
        """
        Generate summary from audio (inference mode)
        
        Args:
            audio_features: Audio features [batch, max_seq_len, feat_dim]
            audio_lengths: Actual audio lengths [batch]
            max_length: Maximum generation length
            
        Returns:
            generated_ids: Generated token ids [batch, seq_len]
        """
        self.eval()
        with torch.no_grad():
            result = self.forward(
                audio_features=audio_features,
                audio_lengths=audio_lengths,
                target_ids=None
            )
            return result["logits"]


