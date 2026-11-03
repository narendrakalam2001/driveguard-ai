-- schema.sql
-- DriveGuard AI violation logging database (SQLite).
-- Applied automatically by src/alert_system.py on first run; kept here
-- as the single source of truth for the schema and for manual inspection.

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------
-- cameras: one row per physical CCTV camera / junction
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cameras (
    camera_id           TEXT PRIMARY KEY,
    location_name       TEXT NOT NULL,
    speed_limit         INTEGER NOT NULL,
    calibration_matrix  TEXT,                 -- JSON-serialized 3x3 homography
    status              TEXT NOT NULL DEFAULT 'ACTIVE'
                             CHECK (status IN ('ACTIVE', 'OFFLINE', 'MAINTENANCE')),
    last_active         TEXT                  -- ISO-8601 UTC timestamp
);

-- ---------------------------------------------------------------------
-- vehicles: one row per tracked vehicle session (per camera session)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS vehicles (
    track_id            INTEGER NOT NULL,
    session_id          TEXT NOT NULL,
    first_seen          TEXT NOT NULL,        -- ISO-8601 UTC timestamp
    last_seen           TEXT NOT NULL,
    avg_speed           REAL,
    max_speed           REAL,
    plate_number        TEXT,
    vehicle_type        TEXT,
    violation_count      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (track_id, session_id)
);

-- ---------------------------------------------------------------------
-- violations: one row per classified violation event
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS violations (
    id                   TEXT PRIMARY KEY,     -- UUID
    timestamp            TEXT NOT NULL,        -- ISO-8601 UTC timestamp
    plate_number         TEXT,
    plate_confidence     REAL,
    vehicle_type         TEXT NOT NULL,
    detected_speed       REAL NOT NULL,
    speed_limit          INTEGER NOT NULL,
    excess_speed         REAL NOT NULL,
    severity             TEXT NOT NULL
                              CHECK (severity IN ('NORMAL', 'WARNING', 'VIOLATION', 'SEVERE')),
    location              TEXT NOT NULL,
    camera_id             TEXT NOT NULL,
    evidence_frame_path   TEXT,
    notice_generated       INTEGER NOT NULL DEFAULT 0,   -- boolean 0/1
    notice_text            TEXT,
    status                  TEXT NOT NULL DEFAULT 'PENDING'
                                 CHECK (status IN ('PENDING', 'RESOLVED', 'DISPUTED')),
    created_at              TEXT NOT NULL,
    FOREIGN KEY (camera_id) REFERENCES cameras (camera_id)
);

CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON violations (timestamp);
CREATE INDEX IF NOT EXISTS idx_violations_plate ON violations (plate_number);
CREATE INDEX IF NOT EXISTS idx_violations_severity ON violations (severity);
CREATE INDEX IF NOT EXISTS idx_violations_camera ON violations (camera_id);
CREATE INDEX IF NOT EXISTS idx_vehicles_plate ON vehicles (plate_number);

-- ---------------------------------------------------------------------
-- rto_dispatches: audit trail of "Send to RTO" actions (currently a mock integration)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rto_dispatches (
    id            TEXT PRIMARY KEY,
    violation_id  TEXT NOT NULL,
    sent_at       TEXT NOT NULL,          -- ISO-8601 UTC timestamp
    channel       TEXT NOT NULL,          -- e.g. 'RTO_PORTAL'
    reference     TEXT NOT NULL,          -- reference returned by the (mock) portal
    mode          TEXT NOT NULL DEFAULT 'mock' CHECK (mode IN ('mock', 'live')),
    FOREIGN KEY (violation_id) REFERENCES violations (id)
);
CREATE INDEX IF NOT EXISTS idx_rto_violation ON rto_dispatches (violation_id);
