# Specification: Auto LRC Tool

## 1. Scanner Workflow
- Quét đệ quy thư mục Walkman tìm: `.mp3`, `.flac`, `.m4a`, `.wav`.
- Đã có `.lrc`: Bỏ qua (Skip).
- Có `.txt`, chưa có `.lrc`: Chuyển thẳng sang bước đối chiếu và ép nhịp.
- Chưa có cả `.lrc` lẫn `.txt`: Gọi Genius Fetcher.

## 2. Genius Fetcher & Interactive Resolution
- Trích xuất Artist/Title qua Mutagen (fallback về tên file sạch).
- Tìm kiếm qua Genius API (thư viện `lyricsgenius`).
- Tìm thấy: Hiển thị 4 câu đầu, yêu cầu xác nhận `[Y]/[N]/[R]`.
- Không thấy hoặc chọn [N]/[R]: Prompt người dùng nhập tên bài hát thay thế (xử lý bài unreleased) để tìm lại.
- Thất bại toàn bộ: Hỏi ý kiến người dùng trước khi chuyển sang tự sinh lời hoàn toàn bằng Whisper.

## 3. Audio Cross-Matching & Fuzzy Refinement
- Dùng `faster-whisper` nghe audio để tạo bản transcript thô.
- Dùng `rapidfuzz` so khớp từng dòng giữa text Genius/txt và Whisper transcript.
- Thuật toán ưu tiên: Nếu âm thanh Whisper khớp ngữ âm với Genius (ví dụ: Whisper nghe "straight up", Genius ghi "STR8 UP!"), giữ nguyên từ ngữ và định dạng dòng của Genius.
- In ra bảng Diff demo trên console để người dùng kiểm chứng trước khi căn chỉnh.

## 4. Forced Alignment & Output
- Dùng module Forced Alignment của `stable-ts` (`model.align`) ép nhịp audio với văn bản đã tinh chỉnh.
- Xuất file `.lrc` chuẩn: `[mm:ss.xx] Lời bài hát`.