"""
Diagnostic: inspect calibration.bin format
Run this first, paste the output, and we'll fix the parser.
"""
from pathlib import Path
import struct
import numpy as np

calib_path = Path(r"D:\Data\Temp-Neon\2026-05-11-17-27-07\calibration.bin")
data = calib_path.read_bytes()

print(f"File size: {len(data)} bytes")
print(f"\nFirst 64 bytes (hex):")
print(data[:64].hex())
print(f"\nFirst 64 bytes (raw, replacing non-printable):")
printable = bytes(b if 32 <= b < 127 else ord('.') for b in data[:64])
print(printable.decode('ascii'))

# Try different integer sizes for a length prefix
print("\n── Possible length prefix interpretations ──")
for fmt, label in [("<H","uint16 LE"), (">H","uint16 BE"),
                   ("<I","uint32 LE"), (">I","uint32 BE"),
                   ("<Q","uint64 LE"), (">Q","uint64 BE")]:
    sz = struct.calcsize(fmt)
    val = struct.unpack_from(fmt, data, 0)[0]
    print(f"  {label}: {val}  (would end at byte {sz + val})")

# Check if it starts with JSON directly
print("\n── First 200 bytes as UTF-8 (ignoring errors) ──")
print(data[:200].decode("utf-8", errors="replace"))

# Try protobuf varint at byte 0
print("\n── Try reading as protobuf-style varint ──")
b0 = data[0]
print(f"  Byte 0: {b0} (0x{b0:02x})")
b1 = data[1]
print(f"  Byte 1: {b1} (0x{b1:02x})")

# Check info.json for calibration hints
info_path = Path(r"D:\Data\Temp-Neon\2026-05-11-17-27-07\info.json")
if info_path.exists():
    print("\n── info.json contents ──")
    print(info_path.read_text())
