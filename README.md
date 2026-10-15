# Day 27 — HITL PR Review Agent

Lab xây agent review pull request bằng LangGraph: lấy diff, tạo phân tích có cấu trúc, định tuyến theo confidence, dừng chờ người duyệt khi cần và ghi lịch sử quyết định vào SQLite.

## Luồng và ngưỡng

Các ngưỡng lấy từ `common/schemas.py`:

| Confidence | Nhánh | Hành vi |
|---|---|---|
| `>= 0.73` | `auto_approve` | Agent đăng comment review lên PR |
| `0.58–<0.73` | `human_approval` | Reviewer approve, reject hoặc yêu cầu sửa |
| `< 0.58` | `escalate` | Reviewer trả lời câu hỏi; agent tổng hợp và chờ duyệt |

Confidence là tự đánh giá về mức đầy đủ, đúng đắn của review, không phải xác suất PR an toàn. Mọi nội dung từ title và diff đều được coi là dữ liệu cần review, không phải lệnh cho agent.

## Cài đặt

Yêu cầu Python 3.11+, Git và `uv`.

```bash
uv sync
cp .env.example .env
```

Điền `.env` bằng API key từ OpenAI, GitHub PAT và username GitHub:

```dotenv
OPENROUTER_API_KEY=sk-proj-your-openai-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
GITHUB_TOKEN=github_pat_or_ghp_token
GITHUB_USER=your_github_username
```

`OPENROUTER_API_KEY` là tên biến legacy mà `common/llm.py` đọc; giá trị có thể là OpenAI API key. Không commit `.env`. GitHub token cần quyền đọc PR và đăng comment trên repo đích. Với PR mẫu public, dùng quyền `public_repo` như hướng dẫn của lab.

## Chạy các exercise

```bash
uv run python exercises/exercise_1_confidence.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/1
uv run python exercises/exercise_1_confidence.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/2
uv run python exercises/exercise_2_hitl.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/1
uv run python exercises/exercise_3_escalation.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/2
uv run python exercises/exercise_4_audit.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/1
uv run streamlit run app.py
```

Trong Streamlit, bấm **Review PR #1** hoặc **Review PR #2** để chạy ngay PR mẫu; nút **Xem PR** bên dưới mở diff trên GitHub. Mặc định giao diện bật **Chạy thử**, nên không đăng comment. Tắt tùy chọn này khi muốn demo đăng comment thật; nhánh auto có thể đăng ngay sau khi phân tích. Form PR URL vẫn dùng được cho PR khác.

Chọn Approve trên PR mẫu sẽ đăng comment công khai. Nhánh escalation tổng hợp review rồi hiển thị yêu cầu duyệt. Edit yêu cầu hướng dẫn, tạo review mới và hỏi reviewer duyệt lại. Reject không đăng comment.

Exercise 1 chỉ minh họa routing. Exercise 2–4 và Streamlit có thể đăng comment ở nhánh auto hoặc sau khi reviewer approve.

## Checkpoint và audit

Checkpointer của LangGraph lưu state để resume; bảng `audit_events` lưu các cột có thể truy vấn. Cả hai ở `hitl_audit.db`. Có thể dùng `HITL_DB_PATH` để đặt file database khác.

```bash
uv run python -m audit.replay --thread THREAD_ID
uv run python -m audit.replay --list
uv run python exercises/exercise_4_audit.py --pr PR_URL --thread THREAD_ID
```

Node có `interrupt()` được chạy lại từ đầu khi resume. Vì thế audit event `pending` trước interrupt có thể xuất hiện lặp. Thời gian trong audit đo xử lý node, không đo thời gian reviewer trả lời. `--thread` tiếp tục checkpoint còn chờ hoặc in kết quả nếu thread đã hoàn tất.

Comment chứa marker theo thread và phiên bản review. Trước khi đăng, agent dò marker đó để nhận ra comment đã tồn tại. POST không được retry tự động; khi mạng đứt ngay sau khi GitHub nhận comment, chạy lại sẽ dò trước rồi trả về comment cũ.

## Bonus trong Streamlit

- **Checkpoint history:** xem các checkpoint và fork phiên approval đang chờ thành thread mô phỏng mới; thread mô phỏng không gọi GitHub.
- **Confidence statistics:** đếm phiên ban đầu, phân bố route, lựa chọn cuối theo bucket confidence và tổng thời gian graph. Đây là thống kê quan sát, không chứng minh confidence đã calibration.
- **Auto-edit:** reviewer yêu cầu thay đổi, LLM tạo phiên bản mới và chờ một quyết định mới.

## Kiểm thử

Các kiểm thử dùng mock cho LLM và GitHub cùng SQLite tạm, không cần gửi request tới OpenAI hoặc đăng comment thật.

```bash
uv run python -m unittest discover -s tests -v
```

Các tình huống bao gồm biên ngưỡng, ba nhánh, interrupt/resume sau khi mở lại checkpointer, approve/reject/edit, escalation và duyệt refined, audit/replay, thống kê, fork mô phỏng, lỗi API và chống đăng trùng. Streamlit được kiểm tra bằng `AppTest` cho trang và tương tác form.

## Nộp bài

Giữ nguyên cấu trúc `common/`, `audit/`, `exercises/` và `app.py`. Repo cần public, đúng mẫu tên `KX-DAY27-HoVaTen-MSSV`, và chạy được sau clone sạch cùng `uv sync`. Trước khi commit, kiểm tra `git status --short`; `.env`, database và file WAL/SHM không được đưa vào Git.
