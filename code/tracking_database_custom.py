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
        import van_utils as lib

        debug = (idx == 2853)

        if debug:
            print("update_tracks input temporal_matches:", len(temporal_matches))
            added = 0
            skipped_obs = 0
            reused_track = 0
            created_track = 0

        for match in temporal_matches:
            prev_feature = match.queryIdx
            curr_feature = match.trainIdx

            prev_obs = lib.get_feature_observation(prev_data, prev_feature, prev_stereo)
            curr_obs = lib.get_feature_observation(curr_data, curr_feature, curr_stereo)

            if prev_obs is None or curr_obs is None:
                if debug:
                    skipped_obs += 1
                continue

            if self.has_feature(idx - 1, prev_feature):
                track_id = self.get_track_of_feature(idx - 1, prev_feature)
                if debug:
                    reused_track += 1
            else:
                track_id = self.create_track()
                lib.add_feature_to_db(self, idx - 1, prev_feature, track_id, prev_obs)
                if debug:
                    created_track += 1

            lib.add_feature_to_db(self, idx, curr_feature, track_id, curr_obs)

            if debug:
                added += 1

        if debug:
            print("update_tracks skipped missing stereo obs:", skipped_obs)
            print("update_tracks reused existing tracks:", reused_track)
            print("update_tracks created new tracks:", created_track)
            print("update_tracks added current observations:", added)
            print(
                "shared tracks after update:",
                len(set(self.tracks(2852)) & set(self.tracks(2853)))
            )

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