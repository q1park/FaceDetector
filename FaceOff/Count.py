from FaceOff import get_frame_idx, get_actor_ids, group_frames_by_actors, remove_outliers_from_snippet, create_snippets

from glob import glob


def count_faces(folder: str, label: str) -> int:
    """Count faces in a folder of images.
    """
    count = 0
    fnames = sorted(glob(folder + '/' + label + '/*'))  # list of images fname
    collection_by_actor = group_frames_by_actors(fnames)

    # then evaluate for each actor
    for k in collection_by_actor.keys():
        print(">> evaluating actor", k)
        videos_by_id = collection_by_actor[k]  # all videos by actor id

        # convert to snippets
        snippets = create_snippets(videos_by_id)  # snippets without outliers removed

        # for each snippet, remove outliers
        for snippet in snippets:
            processed_snippet = remove_outliers_from_snippet(snippet)

            final_snippets = create_snippets(processed_snippet)  # recreate snippets now that outliers have been removed

            for final in final_snippets:
                if len(final) > 10:
                    count += 1    
    return count