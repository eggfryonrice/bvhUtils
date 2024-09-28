import numpy as np


# create new BVH file by data, while having same joint orientation with original file
def createBVH(originalFilePath: str, filePath: str, datas: np.ndarray) -> None:
    with open(originalFilePath, "r") as of:
        with open(filePath, "w") as f:
            # write lines in upper part to new file
            for l in of:
                # edit number of frames
                tokens: list[str] = l.strip().split()
                if tokens[0] == "Frames:":
                    f.write(f"Frames: {len(datas)}\n")
                # after frametime, we move to next part
                elif tokens[0] == "Frame":
                    f.write(l)
                    break
                else:
                    f.write(l)

            # write the data only when time is between t1 and t2
            for data in datas:
                for x in data:
                    f.write(f"{x} ")
                f.write("\n")
