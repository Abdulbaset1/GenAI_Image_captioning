import torch
import numpy as np

def greedy_search(model, image_feature, idx2word, word2idx, device, max_len=20):
    """
    Generate caption using greedy search
    """
    START_IDX = word2idx["<start>"]
    END_IDX = word2idx["<end>"]
    
    model.eval()
    with torch.no_grad():
        # Convert image feature to tensor
        if not isinstance(image_feature, torch.Tensor):
            image_feature = torch.tensor(image_feature, dtype=torch.float32)
        
        feature = image_feature.unsqueeze(0).to(device)
        
        # Get hidden state from encoder
        hidden = model.encoder(feature)
        
        # Start with <start> token
        seq = [START_IDX]
        
        for _ in range(max_len):
            # Prepare input
            inp = torch.tensor(seq).unsqueeze(0).to(device)
            
            # Get predictions
            out = model.decoder(inp, hidden)
            next_word = out[:, -1].argmax(-1).item()
            
            # Stop if <end> token is generated
            if next_word == END_IDX:
                break
                
            seq.append(next_word)
        
        # Convert indices to words
        caption_words = []
        for idx in seq[1:]:  # Skip <start> token
            if idx == END_IDX:
                break
            caption_words.append(idx2word.get(idx, "<unk>"))
        
        return " ".join(caption_words)

def beam_search(model, image_feature, idx2word, word2idx, device, beam_width=3, max_len=20):
    """
    Generate caption using beam search
    """
    START_IDX = word2idx["<start>"]
    END_IDX = word2idx["<end>"]
    
    model.eval()
    with torch.no_grad():
        # Convert image feature to tensor
        if not isinstance(image_feature, torch.Tensor):
            image_feature = torch.tensor(image_feature, dtype=torch.float32)
        
        feature = image_feature.unsqueeze(0).to(device)
        hidden = model.encoder(feature)
        
        # Initialize beams
        beams = [([START_IDX], 0.0)]
        
        for _ in range(max_len):
            new_beams = []
            
            for seq, score in beams:
                # Stop if sequence ended
                if seq[-1] == END_IDX:
                    new_beams.append((seq, score))
                    continue
                
                # Get predictions
                inp = torch.tensor(seq).unsqueeze(0).to(device)
                out = model.decoder(inp, hidden)
                
                # Get probabilities
                probs = torch.softmax(out[:, -1], dim=-1)
                
                # Get top-k words
                topk_probs, topk_indices = torch.topk(probs, beam_width)
                
                for i in range(beam_width):
                    w = topk_indices[0][i].item()
                    p = topk_probs[0][i].item()
                    
                    # Avoid -inf log
                    log_prob = -np.log(p + 1e-10)
                    new_beams.append((seq + [w], score + log_prob))
            
            # Select top beams
            beams = sorted(new_beams, key=lambda x: x[1])[:beam_width]
        
        # Get best sequence
        best_seq = beams[0][0]
        
        # Convert to words
        caption_words = []
        for idx in best_seq[1:]:  # Skip <start>
            if idx == END_IDX:
                break
            caption_words.append(idx2word.get(idx, "<unk>"))
        
        return " ".join(caption_words)
