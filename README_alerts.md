# Alarm Management System - Changes Documentation

This document describes all changes made to implement a persistent alarm management system with status tracking (unresolved, resolved, false alarm), snapshot capture, deduplication, and a full management UI.

---

## Overview

Previously, alarms only existed in React state (in-memory). They were lost on page refresh, capped at 100 items, and had no way to acknowledge, resolve, or review them. The new system persists alarms in PostgreSQL, captures a snapshot at detection time, deduplicates repeated detections, and provides a full alarm management page with filtering, bulk actions, and a detail modal.

---

## Files Changed

### 1. `database_manager.py`

**Added import:** `json` (for serializing detection metadata to JSONB).

**New table — `alarms`** (created automatically in `_create_tables`):

| Column               | Type        | Description                                      |
|----------------------|-------------|--------------------------------------------------|
| `id`                 | SERIAL PK   | Auto-incrementing alarm ID                       |
| `camera_id`          | VARCHAR(50) | Which camera triggered the alarm (e.g. "CAM-01") |
| `type`               | VARCHAR(50) | Detection type: `face`, `fire`, `har`, `weapon`  |
| `severity`           | VARCHAR(20) | `low`, `medium`, `high`, `critical`              |
| `status`             | VARCHAR(20) | `unresolved`, `resolved`, `false_alarm`          |
| `description`        | TEXT        | Human-readable alarm message                     |
| `snapshot`           | TEXT        | Base64-encoded JPEG of the frame at detection    |
| `detection_metadata` | JSONB       | Raw detection data (confidence, bbox, class, etc)|
| `created_at`         | TIMESTAMP   | When the alarm was created                       |
| `resolved_at`        | TIMESTAMP   | When the alarm was resolved (nullable)           |
| `resolved_by`        | VARCHAR(255)| Username of who resolved it (nullable)           |
| `notes`              | TEXT        | Operator notes (nullable)                        |

**New indexes:**
- `idx_alarms_status` on `alarms(status)`
- `idx_alarms_type` on `alarms(type)`
- `idx_alarms_severity` on `alarms(severity)`
- `idx_alarms_created_at` on `alarms(created_at)`

**New methods added (Alarm management section):**

- `create_alarm(camera_id, alarm_type, severity, description, snapshot, detection_metadata)` — Inserts a new alarm row and returns its ID.
- `get_recent_alarm(camera_id, alarm_type, cooldown_seconds)` — Finds the most recent unresolved alarm of the same type+camera within a cooldown window (used for deduplication).
- `list_alarms(status, alarm_type, severity, camera_id, limit, offset)` — Returns paginated alarms with optional filters. The list query excludes the `snapshot` column for performance; snapshots are only returned in `get_alarm`.
- `get_alarm(alarm_id)` — Returns a single alarm with all columns including the snapshot.
- `update_alarm(alarm_id, status, notes, resolved_by)` — Updates alarm status and/or notes. Automatically sets `resolved_at` and `resolved_by` when status is `resolved` or `false_alarm`.
- `get_alarm_stats()` — Returns counts: `unresolved`, `critical_unresolved`, `resolved`, `false_alarm`, and `by_type` (breakdown of unresolved alarms by detection type).
- `bulk_update_alarms(alarm_ids, status, resolved_by)` — Bulk-updates alarm statuses using `WHERE id = ANY(...)`.

**Modified method — `get_statistics()`:**
Now additionally queries and returns:
- `unresolved_alarms` — count of alarms with status `unresolved`
- `critical_alarms` — count of unresolved alarms with severity `critical`
- `total_alarms` — total count of all alarms

---

### 2. `app.py`

**New Pydantic models:**
- `UpdateAlarmRequest` — fields: `status` (optional str), `notes` (optional str)
- `BulkUpdateAlarmsRequest` — fields: `alarm_ids` (list of ints), `status` (str)

**New instance attributes (in `__init__`):**
- `self.alarm_cooldowns` — dict tracking the last alarm creation time per `(camera_id, type)` key
- `self.alarm_cooldown_seconds` — set to `30` seconds (deduplication window)
- `self.alarm_lock` — threading lock for thread-safe cooldown checks

**New method — `_create_alarm_if_needed()`:**
Called from the merging thread whenever alarm-worthy detections are found. Logic:
1. Builds a cooldown key from `camera_id:alarm_type`
2. Checks if the same key was used within the last 30 seconds
3. If not, creates the alarm in the database via `self.db.create_alarm()`
4. This prevents alarm fatigue from the same detection firing every frame

**Modified — merging thread (`setup_background_tasks` > `merging_thread`):**
After all detection results are collected and drawn on the frame, but before encoding:
1. Checks if any alarm-worthy detections exist (unknown faces, fire, non-normal HAR, weapons)
2. If so, captures a snapshot of the annotated frame (resized to 400px width, JPEG quality 60) as base64
3. Calls `_create_alarm_if_needed()` for each detection type found:
   - **Face alarms** — created when `name == 'Unknown'`, severity `critical`
   - **Fire alarms** — created for each fire/smoke detection, severity from detection result
   - **HAR alarms** — created for non-normal actions (fighting, vandalism), severity from detection result
   - **Weapon alarms** — created for each weapon detection, severity from detection result
4. Each alarm includes the snapshot and detection metadata (confidence, class, bbox, etc.)

**New API endpoints (added before the WebSocket endpoint):**

| Method | Endpoint                    | Auth     | Description                                          |
|--------|-----------------------------|----------|------------------------------------------------------|
| GET    | `/api/alarms`               | Any user | List alarms with filters: `status`, `type`, `severity`, `camera_id`, `limit`, `offset` |
| GET    | `/api/alarms/stats`         | Any user | Returns alarm count statistics for the dashboard     |
| GET    | `/api/alarms/{alarm_id}`    | Any user | Get single alarm with full details including snapshot|
| PATCH  | `/api/alarms/{alarm_id}`    | Any user | Update alarm status (`unresolved`/`resolved`/`false_alarm`) and/or notes. Automatically records `resolved_by` from the JWT token |
| POST   | `/api/alarms/bulk-update`   | Any user | Bulk update statuses for multiple alarm IDs at once  |

---

### 3. `frontend/src/components/AlarmManagement.js` (NEW FILE)

Complete alarm management page component with:

**Stats cards (top of page):**
- Unresolved count (red)
- Critical unresolved count (orange)
- Resolved count (green)
- False alarm count (gray)
- Auto-refreshes every 10 seconds

**Filter bar:**
- Dropdown for status: All / Unresolved / Resolved / False Alarm
- Dropdown for type: All / Face / Fire / Action / Weapon
- Dropdown for severity: All / Critical / High / Medium / Low
- Bulk action buttons appear when alarms are selected: "Resolve All" and "Mark False Alarm"

**Alarm table:**
- Checkbox column for bulk selection (with select-all toggle)
- Columns: Time, Type (color-coded badge), Severity (with colored dot), Description, Camera, Status, Actions
- Row actions: View details (eye icon), Resolve (green check), Mark false alarm (X icon)
- Scrollable body with max height
- Pagination controls at the bottom

**Detail modal (opened by clicking the eye icon):**
- Snapshot image (the frame captured at detection time)
- Info grid: type, severity, camera, created timestamp
- Description text
- Detection metadata displayed as key-value pairs
- Resolved info (if applicable): resolved_at, resolved_by
- Notes textarea for adding operator notes
- Action buttons: Resolve, False Alarm, Save Notes, Close

---

### 4. `frontend/src/components/Sidebar.js`

**Changes:**
- Added imports: `useState`, `useEffect` from React; `Bell` icon from lucide-react
- Added `API_BASE` constant for fetching alarm stats
- Added `alarms` nav item (id: `alarms`, icon: Bell, label: "Alarms", visible to all users — not admin-only)
- Added `unresolvedCount` state that polls `/api/alarms/stats` every 10 seconds
- Added a red badge next to the "Alarms" label showing the unresolved count (shows "99+" if over 99)

---

### 5. `frontend/src/App.js`

**Changes:**
- Added import: `AlarmManagement` component
- Added route for `activeTab === 'alarms'` that renders `<AlarmManagement />` inside the standard layout wrapper (accessible to all authenticated users, not just admins)

---

### 6. `frontend/src/components/Statistics.js`

**Changes:**
- Added imports: `Bell` and `AlertTriangle` icons from lucide-react
- Added three new stat cards to the statistics grid:
  - **Total Alarms** (indigo) — `stats.total_alarms`
  - **Unresolved Alarms** (red) — `stats.unresolved_alarms`
  - **Critical Alarms** (rose) — `stats.critical_alarms`

---

## How It Works End-to-End

1. **Detection happens** in the processing threads (face, fire, HAR, weapon)
2. **Merging thread** collects results and checks for alarm-worthy detections
3. If found, it **captures a snapshot** of the annotated frame and calls `_create_alarm_if_needed()`
4. The method checks the **30-second deduplication cooldown** — if the same camera+type alarm was created recently, it skips
5. Otherwise, it **inserts the alarm** into PostgreSQL with all metadata
6. The **frontend** polls `/api/alarms/stats` every 10 seconds to update the sidebar badge
7. The **AlarmManagement page** fetches and displays alarms with filtering and pagination
8. Operators can **resolve**, **mark as false alarm**, add **notes**, or **bulk-update** alarms
9. The **Statistics page** shows aggregate alarm counts

## Alarm Deduplication

To prevent alarm fatigue (e.g., 30 fire alarms per second from consecutive frames), the system uses an in-memory cooldown map:
- Key: `"{camera_id}:{alarm_type}"` (e.g., `"CAM-01:fire"`)
- Value: timestamp of last alarm creation
- If less than 30 seconds have passed since the last alarm of the same key, the new alarm is skipped
- This means at most one alarm per type per camera every 30 seconds
