import whisperx
import sys
import gc

# 1. Load mô hình (Đổi sang large-v3 để có dấu câu và giữ nguyên từ tiếng Anh)
model = whisperx.load_model(
    "large-v3",
    device="cuda",
    language="vi",
    compute_type="float16"
)

# 2. Lấy file âm thanh (nhận từ lệnh hoặc mặc định)
if len(sys.argv) > 1:
    audio_file = sys.argv[1]
else:
    audio_file = "Bluevoice.MP3"

print(f"Đang xử lý file: {audio_file}")
audio = whisperx.load_audio(audio_file)

# 3. Tiến hành nhận diện (transcribe)
print("Đang nhận diện giọng nói...")
result = model.transcribe(audio, batch_size=16)

# Sửa lỗi thiếu key 'language' của WhisperX
if "language" not in result:
    result["language"] = "vi"

# (Tùy chọn) 4. Căn chỉnh thời gian cho chính xác từng từ (Align)
print("Đang căn chỉnh thời gian (Aligning)...")
model_a, metadata = whisperx.load_align_model(language_code="vi", device="cuda")
result = whisperx.align(result["segments"], model_a, metadata, audio, device="cuda", return_char_alignments=False)

import os
from whisperx.utils import get_writer

# Lấy tên file gốc (ví dụ: 'Bluevoice' từ 'whisperX/Bluevoice.MP3')
base_name = os.path.splitext(os.path.basename(audio_file))[0]

# Tạo đường dẫn thư mục Upload/<base_name> (ví dụ: Upload/Bluevoice)
output_dir = os.path.join("Upload", base_name)
os.makedirs(output_dir, exist_ok=True)

# 5. Lưu kết quả ra file
print(f"Đang lưu kết quả vào thư mục: {output_dir}")
# Lưu tất cả định dạng: srt, vtt, txt, tsv, json
writer = get_writer("all", output_dir)
writer_args = {
    "max_line_width": 20,
    "max_line_count": 1,
    "highlight_words": False
}
result["language"] = "vi" # Đảm bảo key language tồn tại sau bước align
writer(result, audio_file, writer_args)

print(f"Xong! Các file đã được lưu tại {output_dir}/")