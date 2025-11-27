import torch
# Cần cài đặt: pip install pycocoevalcap
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

class CIDErReward:
    """Lớp Wrapper để tính toán phần thưởng CIDEr-D cho SCST."""
    def __init__(self, vocab, device):
        self.vocab = vocab
        self.device = device
        # Khởi tạo metric CIDEr-D
        self.cider_scorer = Cider()
        self.tokenizer = PTBTokenizer() 
        print("Khởi tạo CIDErReward Scorer thành công.")

    def _decode_batch(self, sequences):
        """Chuyển đổi tensor chỉ số (B, T) sang list các chuỗi string."""
        decoded = []
        for i in range(sequences.size(0)):
            seq = sequences[i]
            words = []
            for idx in seq:
                idx = idx.item()
                if idx == self.vocab.EOS_idx:
                    break
                if idx not in [self.vocab.SOS_idx, self.vocab.PAD_idx]:
                    words.append(self.vocab.idx2word.get(idx, "<UNK>"))
            decoded.append(" ".join(words))
        return decoded

    def compute(self, sampled_seqs, gt_captions_list, beam_size=5):
        """
        Tính toán phần thưởng CIDEr.
        sampled_seqs: (B * k, T) - Tensor chỉ số từ .sample()
        gt_captions_list: (B,) - List các list string (ground truth)
        """
        batch_size = len(gt_captions_list)
        
        # 1. Decode các chuỗi đã lấy mẫu (sampled)
        decoded_sampled = self._decode_batch(sampled_seqs) # (B * k) strings
        
        # 2. Định dạng Ground Truth (GT) - CẦN SỬA Ở ĐÂY
        # Thư viện yêu cầu định dạng: {img_id: [{'caption': '...'}, ...]}
        gts = {}
        for i in range(batch_size):
            # Lỗi cũ: gts[str(i)] = gt_captions_list[i] (Sai vì là list string)
            # Code mới: Bọc mỗi caption vào dict
            gts[str(i)] = [{'caption': c} for c in gt_captions_list[i]]
            
        # 3. Định dạng kết quả đã sinh (Res) - CẦN SỬA Ở ĐÂY
        # Thư viện yêu cầu định dạng: {img_id: [{'caption': '...'}]}
        res = {}
        for i in range(len(decoded_sampled)):
            img_id_key = str(i // beam_size) # ID của ảnh (0, 1, 2...)
            caption = decoded_sampled[i]
            
            if img_id_key not in res:
                res[img_id_key] = []
            
            # Lỗi cũ: res[img_id_key].append(caption)
            # Code mới: Bọc caption vào dict
            res[img_id_key].append({'caption': caption}) 

        # Tokenize (Cần thiết cho CIDEr)
        # Lúc này gts và res đã đúng chuẩn list of dicts mà PTBTokenizer yêu cầu
        gts_tokenized = self.tokenizer.tokenize(gts)
        res_tokenized = self.tokenizer.tokenize(res)
        
        # 4. Tính điểm CIDEr
        scores, _ = self.cider_scorer.compute_score(gts_tokenized, res_tokenized)
        
        # 5. Định dạng lại điểm số
        # scores trả về là numpy array (B * k,)
        # Cần chuyển về tensor (B, k)
        rewards = torch.tensor(scores, dtype=torch.float32).view(batch_size, beam_size).to(self.device)
        return rewards
