from typing import List

from FaceOff import get_frame_idx, get_actor_ids, group_frames_by_actors, remove_outliers_from_snippet, create_snippets

from glob import glob


def count_static_faces(fnames: List[str], threshold: int) -> int:
    """Count number of static faces in a list of fnames.
    """
    if len(fnames) == 0:
        raise ValueError("Empty folder.")

    n_static_faces = 0
    collection_by_actor = group_frames_by_actors(fnames)

    # separate images by actors since there may be multiple actors/faces in a single frame
    for k in collection_by_actor.keys():
        print(">> evaluating actor", k)
        videos_by_id = collection_by_actor[k]  # all videos by actor id

        # convert to snippets, which are clusters of frames
        snippets = create_snippets(videos_by_id)

        # for each snippet, remove outliers (images that are perceptually different)
        for snippet in snippets:
            processed_snippet = remove_outliers_from_snippet(snippet)
            final_snippets = create_snippets(processed_snippet)  # recreate snippets now that outliers have been removed

            for final in final_snippets:
                if len(final) > threshold:
                    print("snippet length:", len(final), "starting with: ", final[0])
                    n_static_faces += 1    
    return n_static_faces