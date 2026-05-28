"""轻量 CSV 日志记录器: 一行一记, 支持新建 / 追加两种模式。"""

import csv
import os


class CSVLogger:
    def __init__(self, path, fieldnames, _write_header: bool = True):
        self.path = path
        self.fieldnames = list(fieldnames)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if _write_header:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.fieldnames)

    @classmethod
    def append(cls, path, fieldnames):
        """追加模式: 文件已存在则跳过写表头。"""
        if not os.path.exists(path):
            return cls(path, fieldnames)
        return cls(path, fieldnames, _write_header=False)

    def log(self, row):
        if len(row) != len(self.fieldnames):
            raise ValueError(
                f"row 长度 {len(row)} 与 fieldnames 长度 {len(self.fieldnames)} 不一致"
            )
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
