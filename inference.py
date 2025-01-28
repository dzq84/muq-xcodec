import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5" 
import math
import torch
import torchaudio
from model.codec import CODEC

def load_model(checkpoint_path, device='cuda'):
    """
    Load the trained CODEC model from checkpoint.
    """
    model = CODEC.load_from_checkpoint(checkpoint_path,map_location='cuda')
    model.eval()
    model.to(device)
    return model

def process_audio_in_chunks(file_path, sample_rate, model, chunk_size, device='cuda'):
    """
    Process audio file in chunks to avoid memory issues.
    
    Args:
        file_path (str): Path to the input audio file.
        sample_rate (int): Target sample rate for the model.
        model (CODEC): Trained CODEC model.
        chunk_size (int): Chunk size in samples.
        device (str): Device to run the model ('cuda' or 'cpu').
    
    Returns:
        torch.Tensor: Original audio tensor.
        torch.Tensor: Reconstructed audio tensor.
    """
    # Load and preprocess the audio
    audio, sr = torchaudio.load(file_path)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        audio = resampler(audio)

    audio = model.preprocess(audio, sample_rate).to(device)
    audio = audio[:,:]

    # Initialize outputs
    original_chunks = []
    reconstructed_chunks = []
    codes_list = []

    # Process audio in chunks
    for start in range(0, audio.shape[-1], chunk_size):
        end = min(start + chunk_size, audio.shape[-1])
        chunk = audio[0:1, start:end].unsqueeze(dim=0)
     
        with torch.no_grad():
            z_a = model.encoder(chunk).permute(0,2,1)
            z_s = model.semantic(chunk, output_hidden_states=True).last_hidden_state
        
            z = torch.cat((z_a, z_s), dim=2)
            z = model.fc_prior(z).permute(0,2,1)
            
            z_q, codes, latents, commitment_loss, codebook_loss = model.quantizer(z)
            z_q = z_q.permute(0,2,1)

            z_a_q = model.fc_post2(z_q).permute(0,2,1)
            reconstructed_chunk = model.decoder(z_a_q)

            codes_list.append(codes)

        original_chunks.append(chunk.cpu())
        reconstructed_chunks.append(reconstructed_chunk.cpu())

    # Concatenate all chunks
    original_audio = torch.cat(original_chunks, dim=-1)
    reconstructed_audio = torch.cat(reconstructed_chunks, dim=-1)
    codes = torch.cat(codes_list, dim=2)  # Concatenate along the time dimension

    return original_audio, reconstructed_audio, codes

def reconstruct_from_codes(codes, model, device='cuda'):
    """
    Reconstruct audio from quantized codes using RVQ's `from_codes`.
    
    Args:
        codes (torch.Tensor): Quantized codes (B x N x T).
        model (CODEC): Trained CODEC model.
        device (str): Device to run the model ('cuda' or 'cpu').
    
    Returns:
        torch.Tensor: Reconstructed continuous representation z_q.
    """
    with torch.no_grad():
        z_q, _, _ = model.quantizer.from_codes(codes.to(device))
        z_q = z_q.permute(0,2,1)
        z_a_q = model.fc_post2(z_q).permute(0,2,1)
        reconstructed_audio = model.decoder(z_a_q)
    return reconstructed_audio.cpu()

def save_audio(audio_tensor, sample_rate, output_path):
    """
    Save audio tensor to file.
    """
    torchaudio.save(output_path, audio_tensor, sample_rate)

if __name__ == "__main__":
    # Paths and parameters
    checkpoint_path = "muq-xcodec-25/last.ckpt"  # Replace with your checkpoint path
    input_audio_path = "test.flac"  # Replace with your input audio file path
    output_original_path = "gt.wav"  # Path to save the original audio
    output_reconstructed_path = "gn-200tps.wav"  # Path to save the reconstructed audio
    output_reconstructed_from_codes_path = "gn-100tps.wav"  # Path to save reconstructed audio from codes
    sample_rate = 24000  # Model's expected sample rate
    chunk_size = 24000*10 # Process 5 seconds at a time

    # Load the model
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = load_model(checkpoint_path, device=device)

    # Process the audio in chunks
    original_audio, reconstructed_audio, codes = process_audio_in_chunks(
        input_audio_path, sample_rate, model, chunk_size, device=device
    )

    # Reconstruct audio from codes
    codes = codes[:,0:4,:]
    reconstructed_from_codes = reconstruct_from_codes(codes, model, device=device)

    # Save the results
    save_audio(original_audio[0], sample_rate, output_original_path)
    save_audio(reconstructed_audio[0], sample_rate, output_reconstructed_path)
    save_audio(reconstructed_from_codes[0], sample_rate, output_reconstructed_from_codes_path)

    print(f"Original audio saved to {output_original_path}")
    print(f"Reconstructed audio saved to {output_reconstructed_path}")
    print(f"Reconstructed audio from codes saved to {output_reconstructed_from_codes_path}")
