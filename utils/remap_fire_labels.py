import os

FIRE_LABEL_PATHS = [
    "data/raw/fire/train/labels",
    "data/raw/fire/valid/labels",
    "data/raw/fire/test/labels"
]

for folder in FIRE_LABEL_PATHS:
    print("Processing:", folder)

    if not os.path.exists(folder):
        print("⚠ Folder not found, skipping:", folder)
        continue

    for file in os.listdir(folder):
        if not file.endswith(".txt"):
            continue

        path = os.path.join(folder, file)

        if not os.path.isfile(path):
            print("⚠ File not found, skipping:", path)
            continue

        try:
            with open(path, "r") as f:
                lines = f.readlines()
        except Exception as e:
            print("⚠ Cannot read file, skipping:", path)
            continue

        new_lines = []

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            parts[0] = "1"  # FORCE FIRE CLASS = 1
            new_lines.append(" ".join(parts))

        try:
            with open(path, "w") as f:
                for l in new_lines:
                    f.write(l + "\n")
        except Exception:
            print("⚠ Cannot write file, skipping:", path)

print("🔥 Fire labels remapped to class 1 (robust mode)")
