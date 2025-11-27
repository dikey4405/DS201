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
        """
        batch_size = len(gt_captions_list)
        
        # 1. Decode các chuỗi đã lấy mẫu (sampled)
        decoded_sampled = self._decode_batch(sampled_seqs) # (B * k) strings
        
        # 2. Định dạng Ground Truth (GT)
        gts = {}
        for i in range(batch_size):
            # Định dạng đúng: {img_id: [{'caption': '...'}, ...]}
            gts[str(i)] = [{'caption': c} for c in gt_captions_list[i]]
            
        # 3. Định dạng kết quả đã sinh (Res)
        res = {}
        for i in range(len(decoded_sampled)):
            img_id_key = str(i // beam_size) 
            caption = decoded_sampled[i]
            
            if img_id_key not in res:
                res[img_id_key] = []
            
            # Định dạng đúng: {img_id: [{'caption': '...'}]}
            res[img_id_key].append({'caption': caption}) 

        # Tokenize 
        gts_tokenized = self.tokenizer.tokenize(gts)
        res_tokenized = self.tokenizer.tokenize(res)
        
        # 4. Tính điểm CIDEr - SỬA LỖI Ở ĐÂY
        # compute_score trả về (average_score, scores_array)
        # Chúng ta cần scores_array (mảng điểm cho từng ảnh) chứ không phải điểm trung bình
        _, scores = self.cider_scorer.compute_score(gts_tokenized, res_tokenized)
        
        # 5. Định dạng lại điểm số
        # scores bây giờ là mảng có kích thước (batch_size * beam_size)
        rewards = torch.tensor(scores, dtype=torch.float32).view(batch_size, beam_size).to(self.device)
        return rewards
