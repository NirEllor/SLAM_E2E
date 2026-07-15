# tracking_database.py

import pickle
from dataclasses import dataclass

# Support both homework/ (van_utils) and project/ (utils.tracking) imports
try:
    # Try project/ first (namespace package import)
    from utils.tracking import get_feature_observation, add_feature_to_db
except ImportError:
    # Fall back to homework/ van_utils
    import homework.van_utils as lib
    get_feature_observation = lib.get_feature_observation
    add_feature_to_db = lib.add_feature_to_db


############################################################################
# DATABASE DEFINITION - Core data structures for feature track storage
############################################################################

@dataclass
class Observation:
    frame_id: int
    feature_idx: int
    x_left: float
    x_right: float
    y: float


class TrackingDB:
    """Stores feature tracks across multiple frames with frame-track associations and observations."""

    def __init__(self):
        """Initialize empty tracking database with no tracks or frames."""
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
        """Returns list of track IDs visible in a given frame."""
        return self.frame_to_tracks.get(frame_id, [])

    def frames(self, track_id):
        """Returns list of frame IDs where a track appears."""
        return self.track_to_frames.get(track_id, [])

    def observation(self, frame_id, track_id):
        """Returns Observation for a feature in a given frame and track."""
        return self.observations.get((frame_id, track_id))

    def track_num(self):
        """Returns total number of tracks in database."""
        return len(self.track_to_frames)

    def frame_num(self):
        """Returns total number of frames in database."""
        return len(self.frame_to_tracks)

    # ==========================
    # Encapsulated Track Ingestion
    # ==========================

    ############################################################################
    # ADDING FRAMES TO DATABASE - Temporal tracking integration
    ############################################################################
    def update_tracks(self, idx, temporal_matches, prev_stereo, curr_stereo, prev_data, curr_data):
        """Ingests temporal feature matches into tracking database, linking features across frames."""

        for match in temporal_matches:
            prev_feature = match.queryIdx
            curr_feature = match.trainIdx

            prev_obs = get_feature_observation(prev_data, prev_feature, prev_stereo)
            curr_obs = get_feature_observation(curr_data, curr_feature, curr_stereo)

            if prev_obs is None or curr_obs is None:
                continue

            if self.has_feature(idx - 1, prev_feature):
                track_id = self.get_track_of_feature(idx - 1, prev_feature)
            else:
                track_id = self.create_track()
                add_feature_to_db(self, idx - 1, prev_feature, track_id, prev_obs)


            add_feature_to_db(self, idx, curr_feature, track_id, curr_obs)


    # ==========================
    # construction
    # ==========================

    def create_track(self):
        """Creates a new track and returns its ID."""
        track_id = self.next_track_id
        self.next_track_id += 1
        self.track_to_frames[track_id] = []
        return track_id

    def add_observation(self, frame_id, feature_idx, track_id, x_left, x_right, y):
        """Adds a stereo observation (feature in frame) to a track."""
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
        """Checks if a feature has been tracked in a given frame."""
        return (frame_id, feature_idx) in self.feature_to_track

    def get_track_of_feature(self, frame_id, feature_idx):
        """Returns track ID for a feature in a given frame, or None if not tracked."""
        return self.feature_to_track.get((frame_id, feature_idx), None)

    # ==========================
    # serialization
    # ==========================

    def save(self, filename):
        """Serializes tracking database to pickle file."""
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Deserializes tracking database from pickle file."""
        with open(filename, "rb") as f:
            return pickle.load(f)