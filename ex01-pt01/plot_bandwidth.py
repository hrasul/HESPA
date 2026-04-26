import re
import matplotlib.pyplot as plt

CPU_RESULTS_FILE = "bandwidth_measurements.txt"
CUDA_RESULTS_FILE = "bandwidth_cuda_results.txt"
OUTPUT_PLOT = "bandwidth_comparison.png"

# best CUDA bandwidth per nx across all block sizes
# integer, e.g. 256 -> use only that block size

# Best CUDA block size chosen per nx:
#    32 KB: 128
#   256 KB: 128
#     1 MB: 128
#     8 MB: 1024
#    64 MB: 1024
#   128 MB: 1024
#   256 MB: 1024

CUDA_BLOCKSIZE = None

NX_ORDER = [4096, 32768, 131072, 1048576, 8388608, 16777216, 33554432]
BUFFER_LABELS = ["32 KB", "256 KB", "1 MB", "8 MB", "64 MB", "128 MB", "256 MB"]


def parse_cpu_summary(filename):
    serial = []
    parallel = []

    in_summary = False
    with open(filename, "r") as f:
        for raw in f:
            line = raw.strip()

            if "Summary Statistics" in line:
                in_summary = True
                continue
            if not in_summary:
                continue
            if not line or line.startswith("=") or line.startswith("Buffer Size"):
                continue

            # Example:
            # 0 |      28.6954 GB/s |         3.7605 GB/s |     .13x
            parts = [p.strip() for p in raw.split("|")]
            if len(parts) < 4:
                continue

            try:
                s = float(parts[1].replace("GB/s", "").strip())
                p = float(parts[2].replace("GB/s", "").strip())
            except ValueError:
                continue

            serial.append(s)
            parallel.append(p)

    if len(serial) != len(BUFFER_LABELS):
        raise ValueError(
            f"Expected {len(BUFFER_LABELS)} CPU summary rows, found {len(serial)}."
        )

    return serial, parallel


def parse_cuda_summary(filename, selected_blocksize=None):
    data = {}
    in_summary = False

    with open(filename, "r") as f:
        for raw in f:
            line = raw.strip()

            if line.startswith("Average Bandwidth"):
                in_summary = True
                continue
            if not in_summary:
                continue
            if not line or line.startswith("-") or line.startswith("nx"):
                continue

            # Example:
            # 1048576    | 256       | 376.4546
            parts = [p.strip() for p in raw.split("|")]
            if len(parts) < 3:
                continue

            try:
                nx = int(parts[0])
                bs = int(parts[1])
                bw = float(parts[2])
            except ValueError:
                continue

            data[(nx, bs)] = bw

    if not data:
        raise ValueError("No CUDA average summary rows found.")

    cuda_bw = []
    used_bs = []

    for nx in NX_ORDER:
        candidates = [(bs, bw) for (n, bs), bw in data.items() if n == nx]
        if not candidates:
            raise ValueError(f"No CUDA data found for nx={nx}")

        if selected_blocksize is None:
            bs, bw = max(candidates, key=lambda x: x[1])
        else:
            matches = [(bs, bw) for bs, bw in candidates if bs == selected_blocksize]
            if not matches:
                raise ValueError(f"No CUDA data for nx={nx}, blockSize={selected_blocksize}")
            bs, bw = matches[0]

        cuda_bw.append(bw)
        used_bs.append(bs)

    return cuda_bw, used_bs


def main():
    serial_bw, omp_bw = parse_cpu_summary(CPU_RESULTS_FILE)
    cuda_bw, used_bs = parse_cuda_summary(CUDA_RESULTS_FILE, CUDA_BLOCKSIZE)

    x = list(range(len(BUFFER_LABELS)))

    plt.figure(figsize=(10, 6))
    plt.plot(x, serial_bw, marker="o", linewidth=2, label="CPU Serial")
    plt.plot(x, omp_bw, marker="s", linewidth=2, label="CPU OpenMP")

    if CUDA_BLOCKSIZE is None:
        cuda_label = "CUDA (best block size)"
    else:
        cuda_label = f"CUDA (block size = {CUDA_BLOCKSIZE})"

    plt.plot(x, cuda_bw, marker="^", linewidth=2, label=cuda_label)

    plt.xticks(x, BUFFER_LABELS)
    plt.xlabel("Buffer Size")
    plt.ylabel("Bandwidth (GB/s)")
    plt.title("STREAM Bandwidth Comparison")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=200)
    plt.show()

    print(f"Saved plot to {OUTPUT_PLOT}")
    if CUDA_BLOCKSIZE is None:
        print("Best CUDA block size chosen per nx:")
        for label, bs in zip(BUFFER_LABELS, used_bs):
            print(f"  {label:>6}: {bs}")


if __name__ == "__main__":
    main()