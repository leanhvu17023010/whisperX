import os
import gc
import uuid
import shutil
import zipfile
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import whisperx

# Thư mục tạm thời để xử lý file
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_transcriptions")
os.makedirs(TEMP_DIR, exist_ok=True)

# Khai báo biến global cho các mô hình
model = None
model_a = None
metadata = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Quản lý vòng đời ứng dụng: Load các mô hình AI khi khởi động
    và giải phóng bộ nhớ khi tắt server.
    """
    global model, model_a, metadata
    print("=== Đang tải mô hình WhisperX large-v3 vào GPU (CUDA)... ===")
    try:
        # 1. Load mô hình transcribe chính (large-v3)
        model = whisperx.load_model(
            "large-v3",
            device="cuda",
            language="vi",
            compute_type="float16"
        )
        # 2. Load mô hình align căn chỉnh thời gian cho tiếng Việt
        model_a, metadata = whisperx.load_align_model(
            language_code="vi", 
            device="cuda"
        )
        print("=== Tải mô hình thành công! API đã sẵn sàng xử lý. ===")
    except Exception as e:
        print(f"Lỗi tải mô hình: {e}")
        raise e
        
    yield
    
    # Giải phóng VRAM khi tắt ứng dụng
    print("=== Đang giải phóng bộ nhớ và tắt server... ===")
    if model:
        del model
    if model_a:
        del model_a
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(
    title="WhisperX Transcription API",
    description="API chuyển đổi giọng nói tiếng Việt thành văn bản và file phụ đề SRT dùng WhisperX.",
    version="1.0.0",
    lifespan=lifespan
)

def remove_temp_directory(directory_path: str):
    """Xóa thư mục tạm thời sau khi đã gửi file thành công cho client."""
    if os.path.exists(directory_path):
        try:
            shutil.rmtree(directory_path)
            print(f"Đã dọn dẹp thư mục tạm: {directory_path}")
        except Exception as e:
            print(f"Lỗi khi dọn dẹp thư mục tạm {directory_path}: {e}")

@app.get("/health", tags=["System"])
async def health_check():
    """Kiểm tra trạng thái hoạt động của API và kết nối GPU (CUDA)."""
    import torch
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    return {
        "status": "healthy",
        "gpu_available": cuda_available,
        "gpu_device": device_name,
        "models_loaded": model is not None and model_a is not None
    }

@app.post("/transcribe", tags=["Transcription"])
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="File âm thanh hoặc video cần nhận dạng (mp3, wav, m4a, mp4,...)")
):
    """
    Nhận file âm thanh, tiến hành nhận dạng và trả về file ZIP chứa cả file .txt và .srt.
    Tự động dọn dẹp file tạm trên server sau khi gửi xong.
    """
    if not model or not model_a:
        raise HTTPException(
            status_code=503, 
            detail="Các mô hình WhisperX chưa được tải xong hoặc đã xảy ra lỗi khởi động."
        )

    # 1. Tạo thư mục làm việc riêng biệt (unique) cho request này để tránh xung đột đa luồng
    request_id = str(uuid.uuid4())
    request_dir = os.path.join(TEMP_DIR, request_id)
    os.makedirs(request_dir, exist_ok=True)

    # Lấy tên file gốc và phần mở rộng
    original_filename = file.filename
    _, ext = os.path.splitext(original_filename)
    if not ext:
        ext = ".mp3" # Mặc định nếu không lấy được ext

    # Đường dẫn lưu file âm thanh tạm thời
    temp_audio_path = os.path.join(request_dir, f"input{ext}")

    try:
        # 2. Đảm bảo con trỏ file ở đầu và ghi file upload vào đĩa tạm thời
        await file.seek(0)
        content = await file.read()
        file_size = len(content)
        
        print(f"[{request_id}] Đã nhận file {original_filename}, dung lượng: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")
        
        if file_size == 0:
            raise HTTPException(
                status_code=400,
                detail="File tải lên bị trống (0 bytes). Vui lòng kiểm tra lại cấu hình Postman."
            )
            
        with open(temp_audio_path, "wb") as buffer:
            buffer.write(content)
            
        print(f"[{request_id}] Bắt đầu nhận dạng...")

        # 3. Load audio sử dụng whisperx
        audio = whisperx.load_audio(temp_audio_path)

        # 4. Transcribe (Nhận dạng)
        result = model.transcribe(audio, batch_size=16)
        if "language" not in result:
            result["language"] = "vi"

        # 5. Align (Căn chỉnh thời gian)
        result = whisperx.align(
            result["segments"], 
            model_a, 
            metadata, 
            audio, 
            device="cuda", 
            return_char_alignments=False
        )
        result["language"] = "vi" # Đảm bảo ngôn ngữ được gán đúng

        # 6. Ghi kết quả ra file (.txt và .srt) dùng get_writer của whisperx
        from whisperx.utils import get_writer
        writer = get_writer("all", request_dir)
        writer_args = {
            "max_line_width": 20,
            "max_line_count": 1,
            "highlight_words": False
        }
        # Tên cơ sở cho các file đầu ra
        base_output_name = "transcription"
        writer(result, os.path.join(request_dir, f"{base_output_name}{ext}"), writer_args)

        # 7. Định nghĩa đường dẫn file txt và srt được tạo ra
        txt_file_path = os.path.join(request_dir, f"{base_output_name}.txt")
        srt_file_path = os.path.join(request_dir, f"{base_output_name}.srt")

        if not os.path.exists(txt_file_path) or not os.path.exists(srt_file_path):
            raise HTTPException(
                status_code=500,
                detail="Lỗi trong quá trình tạo file srt hoặc txt từ kết quả nhận dạng."
            )

        # 8. Đóng gói 2 file .txt và .srt vào file ZIP
        zip_filename = f"{os.path.splitext(original_filename)[0]}_transcribed.zip"
        zip_file_path = os.path.join(request_dir, zip_filename)

        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Đổi tên file trong zip cho thân thiện với người dùng
            zipf.write(txt_file_path, arcname=f"{os.path.splitext(original_filename)[0]}.txt")
            zipf.write(srt_file_path, arcname=f"{os.path.splitext(original_filename)[0]}.srt")

        print(f"[{request_id}] Đã tạo xong file zip: {zip_file_path}")

        # 9. Thêm tác vụ nền để xóa thư mục request_dir sau khi gửi file xong
        background_tasks.add_task(remove_temp_directory, request_dir)

        # 10. Trả về file ZIP cho Client
        return FileResponse(
            path=zip_file_path,
            filename=zip_filename,
            media_type="application/zip"
        )

    except Exception as e:
        # Nếu có lỗi, đảm bảo dọn dẹp thư mục tạm ngay lập tức
        remove_temp_directory(request_dir)
        print(f"[{request_id}] Lỗi trong quá trình xử lý: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(e)}")
