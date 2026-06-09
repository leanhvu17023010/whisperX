import whisperx
import sys
import gc
import os
import difflib
from whisperx.utils import get_writer

def align_custom_text(whisper_segments, custom_text):
    """
    Khớp văn bản tùy chỉnh (custom_text) vào các segment thời gian thô do Whisper nhận diện.
    Sử dụng difflib.SequenceMatcher để tự động căn chỉnh từ chính xác nhất.
    """
    # 1. Trích xuất tất cả các từ và index segment tương ứng từ kết quả Whisper
    whisper_words = []
    for seg_idx, seg in enumerate(whisper_segments):
        words = seg['text'].strip().split()
        for w in words:
            # Chuẩn hóa từ để so khớp (viết thường, bỏ ký tự đặc biệt)
            norm_w = "".join(c for c in w.lower() if c.isalnum())
            whisper_words.append((w, norm_w, seg_idx))
            
    # Nếu không có từ nào được nhận diện, trả về segment gốc
    if not whisper_words:
        return whisper_segments

    # 2. Trích xuất các từ từ văn bản tùy chỉnh
    custom_words_raw = custom_text.strip().split()
    custom_words = []
    for w in custom_words_raw:
        norm_w = "".join(c for c in w.lower() if c.isalnum())
        custom_words.append((w, norm_w))

    # 3. So khớp hai chuỗi từ chuẩn hóa
    matcher = difflib.SequenceMatcher(
        None, 
        [w[1] for w in whisper_words], 
        [w[1] for w in custom_words]
    )
    
    # Gán segment index cho từng từ trong văn bản tùy chỉnh
    custom_word_segment_idx = [None] * len(custom_words)
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Khớp hoàn hảo, gán segment tương ứng
            for idx in range(i2 - i1):
                custom_word_segment_idx[j1 + idx] = whisper_words[i1 + idx][2]
        elif tag == 'replace':
            # Thay đổi/mismatch. Gán cho segment của từ Whisper tương ứng gần nhất
            for idx in range(j2 - j1):
                whisper_idx = min(i1 + idx, i2 - 1)
                custom_word_segment_idx[j1 + idx] = whisper_words[whisper_idx][2]
        elif tag == 'insert':
            # Từ mới thêm vào văn bản tùy chỉnh. Gán vào segment trước đó
            prev_seg = whisper_words[i1 - 1][2] if i1 > 0 else 0
            for idx in range(j2 - j1):
                custom_word_segment_idx[j1 + idx] = prev_seg
        elif tag == 'delete':
            # Từ trong Whisper bị lược bỏ trong văn bản tùy chỉnh, không cần xử lý
            pass

    # Xử lý các vị trí None (nếu có) bằng cách kế thừa từ segment liền trước
    current_seg = 0
    for idx in range(len(custom_word_segment_idx)):
        if custom_word_segment_idx[idx] is None:
            custom_word_segment_idx[idx] = current_seg
        else:
            current_seg = custom_word_segment_idx[idx]
            
    # 4. Tái cấu trúc các segments mới với văn bản đã sửa đổi
    new_segments = []
    for i in range(len(whisper_segments)):
        new_segments.append({
            'start': whisper_segments[i]['start'],
            'end': whisper_segments[i]['end'],
            'text': ''
        })
        
    for word_idx, seg_idx in enumerate(custom_word_segment_idx):
        word = custom_words[word_idx][0]
        if new_segments[seg_idx]['text']:
            new_segments[seg_idx]['text'] += ' ' + word
        else:
            new_segments[seg_idx]['text'] = word
            
    # Lọc bỏ các segments trống không chứa từ nào
    new_segments = [seg for seg in new_segments if seg['text'].strip()]
    
    return new_segments

def main():
    # 1. Lấy tham số dòng lệnh
    if len(sys.argv) < 2:
        print("Lỗi: Thiếu tham số truyền vào!")
        print("Cách sử dụng: python phoWhisperText.py <đường_dẫn_audio> [đường_dẫn_file_text]")
        print("Ví dụ: python phoWhisperText.py Bluevoice.MP3 (tự động tìm Bluevoice.txt)")
        print("Hoặc: python phoWhisperText.py Bluevoice.MP3 my_script.txt")
        sys.exit(1)

    audio_file = sys.argv[1]
    if len(sys.argv) >= 3:
        text_file = sys.argv[2]
    else:
        # Tự động tìm file .txt cùng tên với file audio
        text_file = os.path.splitext(audio_file)[0] + ".txt"

    # Kiểm tra sự tồn tại của các file
    if not os.path.exists(audio_file):
        print(f"Lỗi: Không tìm thấy file âm thanh tại: {audio_file}")
        sys.exit(1)

    if not os.path.exists(text_file):
        print(f"Lỗi: Không tìm thấy file văn bản tại: {text_file}")
        sys.exit(1)

    print(f"--- Đang tải văn bản từ file: {text_file} ---")
    with open(text_file, "r", encoding="utf-8") as f:
        custom_text = f.read().strip()

    if not custom_text:
        print("Lỗi: File văn bản trống rỗng!")
        sys.exit(1)

    print(f"Nội dung văn bản gốc nạp vào:\n{custom_text}\n" + "-"*40)

    # 2. Load mô hình Transcribe để lấy mốc thời gian thô
    print("1. Đang tải mô hình Whisper large-v3 để nhận diện mốc thời gian thô...")
    model = whisperx.load_model(
        "large-v3",
        device="cuda",
        language="vi",
        compute_type="float16"
    )

    print(f"2. Đang phân tích mốc thời gian từ file âm thanh: {audio_file}...")
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=16)

    if "language" not in result:
        result["language"] = "vi"

    # Giải phóng VRAM của mô hình Transcribe
    del model
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Khớp văn bản tùy chỉnh với mốc thời gian thô của Whisper
    print("3. Đang tự động khớp văn bản kịch bản của bạn với mốc thời gian thô...")
    aligned_segments = align_custom_text(result["segments"], custom_text)

    # 4. Load mô hình Align để căn chỉnh thời gian chuẩn xác từng từ
    print("4. Đang tải mô hình Align tiếng Việt để căn chỉnh thời gian chi tiết...")
    model_a, metadata = whisperx.load_align_model(language_code="vi", device="cuda")
    
    print("5. Đang tiến hành căn chỉnh thời gian cưỡng bức (Forced Alignment)...")
    aligned_result = whisperx.align(
        aligned_segments, 
        model_a, 
        metadata, 
        audio, 
        device="cuda", 
        return_char_alignments=False
    )

    # Đảm bảo ngôn ngữ được gán đúng
    aligned_result["language"] = "vi"

    # 5. Lưu kết quả ra thư mục Upload
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    output_dir = os.path.join("Upload", f"{base_name}_aligned")
    os.makedirs(output_dir, exist_ok=True)

    print(f"6. Đang lưu kết quả phụ đề khớp chuẩn vào thư mục: {output_dir}")
    writer = get_writer("all", output_dir)
    writer_args = {
        "max_line_width": 20,
        "max_line_count": 1,
        "highlight_words": False
    }
    
    writer(aligned_result, audio_file, writer_args)
    print(f"\n Hoàn thành! Các file phụ đề khớp chuẩn đã được lưu tại: {output_dir}/")

if __name__ == "__main__":
    main()
