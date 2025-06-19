import sys
import math

def main():
    x = []
    y = []

    # Read input lines from stdin
    for line in sys.stdin:
        parts = line.strip().split()
        if len(parts) != 2:
            continue  # skip malformed lines
        try:
            x_val = float(parts[0])
            y_val = float(parts[1])
        except ValueError:
            continue  # skip non-numeric lines
        x.append(x_val)
        y.append(y_val)

    n = len(x)

    if n == 0:
        print(f"Error in xycorr. Number of lines {n}")
        return

    x0 = sum(x) / n
    y0 = sum(y) / n

    t = nx = ny = 0.0
    for i in range(n):
        dx = x[i] - x0
        dy = y[i] - y0
        t += dx * dy
        nx += dx * dx
        ny += dy * dy

    if nx * ny == 0.0:
        c = 0.0
    else:
        c = t / math.sqrt(nx * ny)

    print(f"Pearson coefficient for N= {n} data: {c:8.5f}")

if __name__ == "__main__":
    main()
