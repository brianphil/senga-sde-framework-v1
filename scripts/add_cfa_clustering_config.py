#!/usr/bin/env python3
"""
Add CFA Clustering Configuration Parameters

Run this script to add/update CFA clustering parameters in business_config database.
These parameters control consolidation aggressiveness and route compatibility.

Usage:
    python scripts/add_cfa_clustering_config.py

After running, parameters can be updated via:
    - API: POST /config/business/{key}
    - Database: Direct SQL updates to business_config table
"""

import sqlite3
from pathlib import Path
from datetime import datetime

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DB = PROJECT_ROOT / "data" / "senga_config.db"


def add_cfa_clustering_params():
    """Add CFA clustering parameters to business configuration"""

    if not CONFIG_DB.exists():
        print(f"❌ Config database not found: {CONFIG_DB}")
        print("   Run 'python scripts/initialize_config.py' first")
        return False

    conn = sqlite3.connect(str(CONFIG_DB))
    cursor = conn.cursor()

    timestamp = datetime.now().isoformat()

    # CFA Clustering Parameters
    # These control how aggressively shipments are consolidated
    params = [
        # Tier 1: Close destinations (< 50km) - Always batch if compatible
        (
            "cfa_compatibility_distance_tight",
            "50.0",
            "float",
            "Max distance (km) for tight clustering - always batch if same direction",
            10.0,
            100.0,
        ),
        # Tier 2: Medium distance (50-150km) - Batch if on same highway/collinear
        (
            "cfa_compatibility_distance_medium",
            "150.0",
            "float",
            "Max distance (km) for medium clustering - batch if collinear (e.g., Nakuru-Eldoret)",
            50.0,
            300.0,
        ),
        (
            "cfa_collinearity_threshold_medium",
            "0.85",
            "float",
            "Min collinearity score [0,1] for medium distance batching",
            0.7,
            0.95,
        ),
        (
            "cfa_detour_ratio_threshold_medium",
            "1.3",
            "float",
            "Max detour ratio for medium distance (1.3 = 30% extra distance acceptable)",
            1.0,
            2.0,
        ),
        # Tier 3: Long distance (150-300km) - Batch only if highly collinear
        (
            "cfa_compatibility_distance_long",
            "300.0",
            "float",
            "Max distance (km) for long clustering - batch only if highly collinear (e.g., Nairobi-Kisumu)",
            100.0,
            500.0,
        ),
        (
            "cfa_collinearity_threshold_long",
            "0.92",
            "float",
            "Min collinearity score [0,1] for long distance batching (stricter)",
            0.85,
            0.98,
        ),
        (
            "cfa_detour_ratio_threshold_long",
            "1.15",
            "float",
            "Max detour ratio for long distance (1.15 = 15% extra - stricter)",
            1.0,
            1.5,
        ),
        # Batch size limits
        (
            "cfa_max_batch_size",
            "10",
            "int",
            "Maximum number of shipments per batch",
            2,
            20,
        ),
    ]

    added = 0
    updated = 0

    for key, value, vtype, desc, min_val, max_val in params:
        # Check if parameter exists
        cursor.execute(
            "SELECT parameter_key FROM business_config WHERE parameter_key = ?", (key,)
        )
        exists = cursor.fetchone()

        if exists:
            # Update existing
            cursor.execute(
                """
                UPDATE business_config
                SET parameter_value = ?,
                    value_type = ?,
                    description = ?,
                    min_value = ?,
                    max_value = ?,
                    updated_at = ?,
                    updated_by = ?
                WHERE parameter_key = ?
            """,
                (value, vtype, desc, min_val, max_val, timestamp, "system", key),
            )
            updated += 1
            print(f"   ✓ Updated: {key} = {value}")
        else:
            # Insert new
            cursor.execute(
                """
                INSERT INTO business_config
                (parameter_key, parameter_value, value_type, description, 
                 min_value, max_value, updated_at, updated_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (key, value, vtype, desc, min_val, max_val, timestamp, "system"),
            )
            added += 1
            print(f"   + Added: {key} = {value}")

    conn.commit()
    conn.close()

    print(f"\n✅ CFA clustering configuration updated:")
    print(f"   - {added} parameters added")
    print(f"   - {updated} parameters updated")
    print(f"\n📝 Example adjustments for different business needs:")
    print(f"   • More aggressive consolidation (batch Nairobi-Kisumu):")
    print(f"     - Increase cfa_compatibility_distance_long to 400km")
    print(f"     - Decrease cfa_collinearity_threshold_long to 0.88")
    print(f"   • Conservative (only nearby shipments):")
    print(f"     - Decrease cfa_compatibility_distance_medium to 80km")
    print(f"     - Increase cfa_collinearity_threshold_medium to 0.90")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("CFA Clustering Configuration Setup")
    print("=" * 60)
    print()

    success = add_cfa_clustering_params()

    if success:
        print("\n" + "=" * 60)
        print("Configuration successfully updated!")
        print("=" * 60)
        print("\n💡 To adjust parameters:")
        print("   1. Via API: POST /config/business/{parameter_key}")
        print("   2. Via Database: Edit data/senga_config.db")
        print("   3. System will reload on next CFA initialization")
    else:
        print("\n❌ Configuration update failed")
