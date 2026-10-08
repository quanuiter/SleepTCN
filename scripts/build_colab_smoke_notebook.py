"""Generate the replayable notebook artifact from its reviewed Python cells."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'notebooks/colab_t4_smoke_20261004'
OUT = ROOT / 'runs/colab_t4_smoke_20261004/SleepTCN_T4_Smoke_20261004.ipynb'
cells = [{'cell_type': 'markdown', 'metadata': {}, 'source': [
    '# SleepTCN: kiểm tra GPU T4 trước khi chạy chiến dịch\n',
    'Chỉ đo thiết bị trên 8 mẫu feature nguồn, không huấn luyện fold, không chấm SHHS.\n',
    'Tải `SleepTCN_T4_Smoke_20261004.zip` vào ngăn Tệp trước cell 2.\n',
    'Cell 3 dừng nếu số học không khớp CPU; giới hạn 300 giây mỗi backend.\n',
    'Phiên bản PyTorch khác môi trường gốc phải được ghi nhận; không xem đây là resume tương đương.\n',
    'Cell 4 tải kết quả về máy; runtime tạm có thể bị ngắt.\n']}]
for path in sorted(SOURCE.glob('*.py')):
    cells.append({'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
                  'source': path.read_text(encoding='utf-8').splitlines(keepends=True)})
notebook = {'nbformat': 4, 'nbformat_minor': 4, 'metadata': {
    'colab': {'name': OUT.name}, 'accelerator': 'GPU',
    'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}}, 'cells': cells}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding='utf-8')
assert len(json.loads(OUT.read_bytes())['cells']) == 5
print(OUT)
