"""
Transformer-based models for seismic interpolation.

This module implements transformer-based architectures for integrating
geophone and DAS data for interpolation of missing channels.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model, max_len=5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model (int): Embedding dimension
            max_len (int): Maximum sequence length
        """
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as a buffer (won't be updated during training)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Add positional encoding to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: Input with positional encoding added
        """
        x = x + self.pe[:x.size(1), :]
        return x

class SeismicTransformerEncoder(nn.Module):
    """Transformer encoder for seismic data processing."""
    
    def __init__(self, input_dim, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        """
        Initialize transformer encoder.
        
        Args:
            input_dim (int): Input feature dimension (time steps)
            d_model (int): Model dimension
            nhead (int): Number of attention heads
            num_layers (int): Number of transformer layers
            dim_feedforward (int): Feedforward dimension
            dropout (float): Dropout rate
        """
        super().__init__()
        
        # Input projection to d_model dimension
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Input: (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Forward pass through transformer encoder.
        
        Args:
            src (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            src_mask (torch.Tensor, optional): Mask for self-attention
            src_key_padding_mask (torch.Tensor, optional): Mask for padding tokens
            
        Returns:
            torch.Tensor: Encoded output of shape (batch_size, seq_len, d_model)
        """
        # Project input to d_model dimension
        src = self.input_projection(src)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Apply transformer encoder
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        
        return output

class SeismicTransformerDecoder(nn.Module):
    """Transformer decoder for seismic data interpolation."""
    
    def __init__(self, d_model=512, output_dim=256, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        """
        Initialize transformer decoder.
        
        Args:
            d_model (int): Model dimension
            output_dim (int): Output feature dimension
            nhead (int): Number of attention heads
            num_layers (int): Number of transformer layers
            dim_feedforward (int): Feedforward dimension
            dropout (float): Dropout rate
        """
        super().__init__()
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Input: (batch, seq, feature)
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, output_dim)
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Forward pass through transformer decoder.
        
        Args:
            tgt (torch.Tensor): Target input tensor of shape (batch_size, tgt_len, d_model)
            memory (torch.Tensor): Memory from encoder of shape (batch_size, src_len, d_model)
            tgt_mask (torch.Tensor, optional): Mask for target self-attention
            memory_mask (torch.Tensor, optional): Mask for encoder-decoder attention
            tgt_key_padding_mask (torch.Tensor, optional): Mask for padding in target
            memory_key_padding_mask (torch.Tensor, optional): Mask for padding in memory
            
        Returns:
            torch.Tensor: Decoded output of shape (batch_size, tgt_len, output_dim)
        """
        # Apply transformer decoder
        output = self.transformer_decoder(
            tgt, memory, tgt_mask, memory_mask, 
            tgt_key_padding_mask, memory_key_padding_mask
        )
        
        # Project to output dimension
        output = self.output_projection(output)
        
        return output

class MultimodalSeismicTransformer(nn.Module):
    """
    Multimodal transformer for integrating DAS and geophone data.
    
    This model consists of separate encoders for DAS and geophone data,
    followed by cross-attention and a decoder to reconstruct missing geophone channels.
    """
    
    def __init__(self, time_steps, num_geo_channels, num_das_channels, 
                 d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1):
        """
        Initialize multimodal transformer.
        
        Args:
            time_steps (int): Number of time steps in seismic data
            num_geo_channels (int): Number of geophone channels
            num_das_channels (int): Number of DAS channels
            d_model (int): Model dimension
            nhead (int): Number of attention heads
            num_encoder_layers (int): Number of encoder layers
            num_decoder_layers (int): Number of decoder layers
            dim_feedforward (int): Feedforward dimension
            dropout (float): Dropout rate
        """
        super().__init__()
        
        self.time_steps = time_steps
        self.num_geo_channels = num_geo_channels
        self.num_das_channels = num_das_channels
        self.d_model = d_model
        
        # DAS data encoder
        self.das_encoder = SeismicTransformerEncoder(
            input_dim=time_steps,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Geophone data encoder
        self.geo_encoder = SeismicTransformerEncoder(
            input_dim=time_steps,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Decoder for interpolating missing geophone channels
        self.decoder = SeismicTransformerDecoder(
            d_model=d_model,
            output_dim=time_steps,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Positional embeddings for geophone positions
        self.geo_position_embedding = nn.Embedding(num_geo_channels, d_model)
        
        # Embedding for token type (DAS or geophone)
        self.token_type_embedding = nn.Embedding(2, d_model)  # 0=DAS, 1=geophone
        
    def forward(self, das_data, masked_geo_data, geo_mask=None):
        """
        Forward pass of the multimodal transformer.
        
        Args:
            das_data (torch.Tensor): DAS data of shape (batch_size, num_das_channels, time_steps)
            masked_geo_data (torch.Tensor): Masked geophone data of shape (batch_size, num_geo_channels, time_steps)
            geo_mask (torch.Tensor, optional): Boolean mask for geophone channels (True = masked)
                of shape (batch_size, num_geo_channels)
                
        Returns:
            torch.Tensor: Reconstructed geophone data of shape (batch_size, num_geo_channels, time_steps)
        """
        batch_size = das_data.shape[0]
        
        # Create a key padding mask for masked geophone channels (True = masked)
        if geo_mask is None:
            geo_padding_mask = (masked_geo_data.sum(dim=2) == 0)
        else:
            geo_padding_mask = geo_mask
            
        # Encode DAS data
        # Transpose to (batch_size, num_das_channels, time_steps) -> (batch_size, num_das_channels, d_model)
        das_encoded = self.das_encoder(das_data)
        
        # Encode geophone data (masked)
        # Transpose to (batch_size, num_geo_channels, time_steps) -> (batch_size, num_geo_channels, d_model)
        geo_encoded = self.geo_encoder(masked_geo_data)
        
        # Add token type embeddings
        das_token_type = torch.zeros(batch_size, self.num_das_channels, device=das_data.device).long()
        geo_token_type = torch.ones(batch_size, self.num_geo_channels, device=masked_geo_data.device).long()
        
        das_encoded = das_encoded + self.token_type_embedding(das_token_type).unsqueeze(1)
        geo_encoded = geo_encoded + self.token_type_embedding(geo_token_type).unsqueeze(1)
        
        # Add positional embeddings for geophone channels
        geo_positions = torch.arange(self.num_geo_channels, device=masked_geo_data.device).expand(batch_size, -1)
        geo_encoded = geo_encoded + self.geo_position_embedding(geo_positions).unsqueeze(1)
        
        # Use the decoder to reconstruct missing geophone channels
        # DAS encoding is used as memory, geophone encoding as target input
        reconstructed = self.decoder(
            geo_encoded, 
            das_encoded,
            tgt_key_padding_mask=geo_padding_mask
        )
        
        return reconstructed

class StorSeismicBERTModel(nn.Module):
    """
    StorSeismic BERT-like model for seismic data interpolation.
    
    This model follows the approach of StorSeismic (Harsuko & Alkhalifah, 2022),
    adapting the BERT architecture to seismic data.
    """
    
    def __init__(self, max_channels, time_steps, d_model=768, nhead=12, 
                 num_layers=12, dim_feedforward=3072, dropout=0.1, 
                 pad_token_id=-1, type_vocab_size=2):
        """
        Initialize StorSeismic BERT model.
        
        Args:
            max_channels (int): Maximum number of channels (geophone + DAS)
            time_steps (int): Number of time steps in seismic data
            d_model (int): Model dimension
            nhead (int): Number of attention heads
            num_layers (int): Number of transformer layers
            dim_feedforward (int): Feedforward dimension
            dropout (float): Dropout rate
            pad_token_id (int): ID used for padding tokens
            type_vocab_size (int): Size of vocabulary for token types (e.g., DAS vs geophone)
        """
        super().__init__()
        
        self.max_channels = max_channels
        self.time_steps = time_steps
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        
        # Trace embeddings (transform time series to embeddings)
        self.trace_embedding = nn.Linear(time_steps, d_model)
        
        # Position embeddings for each channel
        self.position_embeddings = nn.Embedding(max_channels, d_model)
        
        # Token type embeddings (DAS or geophone)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, d_model)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection to reconstruct time series
        self.output_projection = nn.Linear(d_model, time_steps)
        
    def forward(self, input_data, attention_mask=None, token_type_ids=None, position_ids=None):
        """
        Forward pass of the StorSeismic BERT model.
        
        Args:
            input_data (torch.Tensor): Input data of shape (batch_size, num_channels, time_steps)
            attention_mask (torch.Tensor, optional): Attention mask of shape (batch_size, num_channels)
                where 1 = attend, 0 = ignore
            token_type_ids (torch.Tensor, optional): Token type IDs of shape (batch_size, num_channels)
                0 = DAS, 1 = geophone
            position_ids (torch.Tensor, optional): Position IDs of shape (batch_size, num_channels)
                
        Returns:
            torch.Tensor: Reconstructed seismic data of shape (batch_size, num_channels, time_steps)
        """
        batch_size, num_channels, _ = input_data.shape
        
        # Create embedding input by transposing input data
        # (batch_size, num_channels, time_steps)
        embedding_input = input_data
        
        # Create attention mask for padding if not provided
        if attention_mask is None:
            # Detect padding: assume pad_token_id means padding
            attention_mask = (input_data.sum(dim=2) != 0).float()
        
        # Create token type IDs if not provided
        if token_type_ids is None:
            token_type_ids = torch.zeros(batch_size, num_channels, device=input_data.device).long()
            
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(num_channels, device=input_data.device).expand(batch_size, -1)
        
        # Convert attention mask for transformer: (1 = attend, 0 = ignore) -> (False = attend, True = ignore)
        transformer_attention_mask = ~(attention_mask.bool())
        
        # Embed the traces
        embeddings = self.trace_embedding(embedding_input)
        
        # Add position embeddings
        embeddings = embeddings + self.position_embeddings(position_ids)
        
        # Add token type embeddings
        embeddings = embeddings + self.token_type_embeddings(token_type_ids)
        
        # Apply layer normalization and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Apply transformer encoder
        # Shape: (batch_size, num_channels, d_model)
        encoded = self.transformer_encoder(embeddings, src_key_padding_mask=transformer_attention_mask)
        
        # Project back to time series
        # Shape: (batch_size, num_channels, time_steps)
        output = self.output_projection(encoded)
        
        return output