import imagehash
from PIL import Image

from typing import List


def get_frame_idx(fname: str) -> int:
    """return frame index from a fname.
    
    for example, "dataset/608832786432738882426817735212/static/00052.00000.jpg" returns "52".
    """
    return int(fname.split("/")[-1].split(".")[0])


def get_actor_ids(fnames: List[str]) -> List[str]:
    """from a list of frames, return a list of actor ids.
    """
    actor_ids = []
    for fname in fnames:
        actor_id = int(fname.split(".")[-2])
        if actor_id not in actor_ids:
            actor_ids.append(actor_id)
    return actor_ids


def group_frames_by_actors(fnames: List[str]) -> dict:
    """group frames by actor id. returns a dict where k = actor id, v = list[str].
    """
    actor_ids = get_actor_ids(fnames)  # get all possible IDs for a list of images
    
    fnames_by_actor = {}
    for actor in actor_ids:
        fnames_by_actor[actor] = []
    
    for actor in actor_ids:
        for fname in fnames:
            actor_id = int(fname.split(".")[-2])
            if actor_id == actor:
                fnames_by_actor[actor].append(fname)
    return fnames_by_actor


def remove_outliers_from_snippet(fnames: List[str]) -> List[str]:
    """Remove outlier images from a list of images.
    """
    stack = []
    for i in range(len(fnames) - 1):
        prev_f = fnames[i]
        next_f = fnames[i + 1]

        prev_h = imagehash.average_hash(Image.open(prev_f))
        next_h = imagehash.average_hash(Image.open(next_f))

        # keep if it's similar
        if next_h == prev_h:
            stack.append(prev_f)
            stack.append(next_f)
        else:
            pass
    return sorted(list(set(stack)))


def create_snippets(video: List[str]) -> List[List[str]]:
    """Generate continguous snippets.
    
    A snippet consists of a sequence of consecutive frames that are separated by no 
    more than MAX_FRAME_DIFF.
    """
    MAX_FRAME_DIFF = 5

    blocks = []
    block = []

    for i in range(len(video) - 1):
        p_idx = get_frame_idx(video[i])
        n_idx = get_frame_idx(video[i + 1])

        if (n_idx - p_idx) < MAX_FRAME_DIFF:
            block.append(video[i])
            block.append(video[i + 1])
        else:
            if len(block) > 0:
                blocks.append(sorted(list(set(block))))
            block = []
    blocks.append(sorted(list(set(block))))
    return blocks