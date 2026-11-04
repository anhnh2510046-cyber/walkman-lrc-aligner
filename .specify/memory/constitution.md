# Constitution: Walkman Auto-LRC Engine

1. Zero Cost & 100% Local: Chỉ sử dụng các thư viện mã nguồn mở chạy local (faster-whisper, stable-ts, rapidfuzz, lyricsgenius, mutagen). Tuyệt đối không dùng API trả phí.
2. Non-destructive: Chỉ tạo file `.lrc` và `.txt` mới, tuyệt đối không chỉnh sửa, nén lại hoặc xóa file audio gốc.
3. Interactive Confirmation: Khi gặp bài hát unreleased hoặc tìm kiếm mơ hồ, bắt buộc dừng lại hỏi người dùng qua console.
4. Canonical Line Integrity: Đối với tiếng lóng hoặc viết tắt, nếu đối chiếu audio xác nhận khớp, ưu tiên giữ nguyên cách viết và ngắt dòng từ Genius.