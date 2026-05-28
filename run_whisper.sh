#!/bin/bash

# Kiểm tra xem người dùng đã truyền tên file vào chưa
if [ "$#" -eq 0 ]; then
    echo "Sử dụng: ./run_whisper.sh <ten_file_am_thanh> [ma_ngon_ngu]"
    echo "Ví dụ 1 (Tự nhận diện ngôn ngữ): ./run_whisper.sh \"Rick Sorkin.wav\""
    echo "Ví dụ 2 (Chỉ định tiếng Việt):    ./run_whisper.sh \"Rick Sorkin.wav\" vi"
    echo "Ví dụ 3 (Chỉ định tiếng Anh):     ./run_whisper.sh \"Rick Sorkin.wav\" en"
    exit 1
fi

INPUT_FILE="$1"
LANGUAGE="$2"

# Kiểm tra xem file có tồn tại không
if [ ! -f "$INPUT_FILE" ]; then
    echo "Lỗi: Không tìm thấy file '$INPUT_FILE'"
    exit 1
fi

# Lấy tên file không bao gồm phần đuôi mở rộng (ví dụ bỏ đi .wav, .mp3)
FILENAME=$(basename -- "$INPUT_FILE")
FILENAME_NO_EXT="${FILENAME%.*}"

OUTPUT_DIR="Results/${FILENAME_NO_EXT}"

echo "=========================================="
echo "Đang xử lý file: $INPUT_FILE"
if [ -n "$LANGUAGE" ]; then
    echo "Ngôn ngữ chỉ định: $LANGUAGE"
else
    echo "Ngôn ngữ: Tự động nhận diện"
fi
echo "Kết quả sẽ được lưu vào: $OUTPUT_DIR"
echo "=========================================="

# Kích hoạt môi trường ảo (vì bạn đang dùng venv)
source venv/bin/activate

# Chạy WhisperX với ngôn ngữ nếu có
if [ -n "$LANGUAGE" ]; then
    whisperx "$INPUT_FILE" --output_dir "$OUTPUT_DIR" --language "$LANGUAGE"
else
    whisperx "$INPUT_FILE" --output_dir "$OUTPUT_DIR"
fi
