# Create dictionary containing names and their labels for dataset
import os
import pickle

curr_dir = os.path.dirname(os.path.abspath(__file__))

def createDict():
    dir = os.path.join(curr_dir, "dataset/train")

    name_label = {}
    name_id = 1
    for dirs in os.listdir(dir):
        if not dirs.startswith("."):
            name_label[dirs] = name_id
            name_id += 1
    return name_label

if os.path.exists(os.path.join(curr_dir,"name_label")):
    if input("name_label dict exists\nrecreate dict? (y==yes): ") == "y":
        name_label = createDict()
        os.remove(os.path.join(curr_dir,"name_label"))
        with open("name_label", "wb") as f:
            pickle.dump(name_label, f)

else:
    name_label = createDict()
    with open("name_label", "wb") as f:
        pickle.dump(name_label, f)
