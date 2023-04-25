from glob import glob

files = []
for file in glob("/home/sheid/Projects/JQF/fuzz-results-runner/corpus/*"):
    with open(file, "rb") as f:
        files.append(f.read())
s = set(files)

if __name__ == '__main__':
    print(f"{len(files)=} {len(s)=}")