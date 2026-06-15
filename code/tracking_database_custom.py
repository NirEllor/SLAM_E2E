# tracking_database_custom.py

import pickle
from dataclasses import dataclass


@dataclass
class Observation:
    frame_id: int
    feature_idx: int
    x_left: float
    x_right: float
    y: float


class TrackingDB:

    def __init__(self):
        self.next_track_id = 0
        
        # track_id -> list(frame_ids)
        self.track_to_frames = {}
        
        # frame_id -> list(track_ids)
        self.frame_to_tracks = {}
        
        # (frame_id, track_id) -> Observation
        self.observations = {}
        
        # (frame_id, feature_idx) -> track_id
        self.feature_to_track = {}
        
        # Array container to hold extra validation metrics safely
        self.inlier_percentages = []

    # ==========================
    # API required by exercise
    # ==========================

    def tracks(self, frame_id):
        return self.frame_to_tracks.get(frame_id, [])

    def frames(self, track_id):
        return self.track_to_frames.get(track_id, [])

    def observation(self, frame_id, track_id):
        return self.observations.get((frame_id, track_id))

    def track_num(self):
        return len(self.track_to_frames)

    def frame_num(self):
        return len(self.frame_to_tracks)

    # ==========================
    # Encapsulated Track Ingestion
    # ==========================

    def update_tracks(self, idx, temporal_matches, prev_stereo, curr_stereo, prev_data, curr_data):
        """
        Atomically ingests frame-to-frame temporal matches and populates internal tracking maps.
        Shields loop indexes and helper extractions entirely from the execution scripts.
        """
        import van_utils as lib
        
        for match in temporal_matches:
            prev_feature = match.queryIdx
            curr_feature = match.trainIdx

            prev_obs = lib.get_feature_observation(prev_data, prev_feature, prev_stereo)
            curr_obs = lib.get_feature_observation(curr_data, curr_feature, curr_stereo)

            if prev_obs is None or curr_obs is None:
                continue

            # Case A: Feature was already registered in a track during the previous frame
            if self.has_feature(idx - 1, prev_feature):
                track_id = self.get_track_of_feature(idx - 1, prev_feature)
            # Case B: New feature sequence detected; initialize a brand-new track
            else:
                track_id = self.create_track()
                lib.add_feature_to_db(self, idx - 1, prev_feature, track_id, prev_obs)

            # Append current frame observation to the running track link
            lib.add_feature_to_db(self, idx, curr_feature, track_id, curr_obs)

    # ==========================
    # construction
    # ==========================

    def create_track(self):
        track_id = self.next_track_id
        self.next_track_id += 1
        self.track_to_frames[track_id] = []
        return track_id

    def add_observation(self, frame_id, feature_idx, track_id, x_left, x_right, y):
        obs = Observation(frame_id, feature_idx, x_left, x_right, y)
        self.observations[(frame_id, track_id)] = obs
        self.feature_to_track[(frame_id, feature_idx)] = track_id

        if frame_id not in self.frame_to_tracks:
            self.frame_to_tracks[frame_id] = []
        self.frame_to_tracks[frame_id].append(track_id)

        if frame_id not in self.track_to_frames[track_id]:
            self.track_to_frames[track_id].append(frame_id)

    # ==========================
    # lookup helpers
    # ==========================

    def has_feature(self, frame_id, feature_idx):
        return (frame_id, feature_idx) in self.feature_to_track

    def get_track_of_feature(self, frame_id, feature_idx):
        return self.feature_to_track.get((frame_id, feature_idx), None)

    # ==========================
    # serialization
    # ==========================

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)