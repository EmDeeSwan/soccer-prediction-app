import psycopg2
import json
from datetime import datetime
import os
from dotenv import load_dotenv

def get_db_schema_as_json(database_url: str, output_filename: str = 'db_schema.json'):
    """
    Standalone script to extract PostgreSQL schema with keys, constraints, and relationships.
    
    Args:
        database_url (str): The PUBLIC Railway connection string
        output_filename (str): Output JSON file name
    """
    conn = None
    try:
        print("🔗 Connecting to Railway PostgreSQL...")
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        print("✅ Connected successfully!")

        # Comprehensive schema query
        schema_query = """
        SELECT
            c.table_name,
            c.column_name,
            c.data_type,
            c.is_nullable,
            c.column_default,
            c.character_maximum_length,
            c.numeric_precision,
            c.numeric_scale,
            c.ordinal_position,
            -- Primary Key Info
            CASE 
                WHEN pk.column_name IS NOT NULL THEN true 
                ELSE false 
            END as is_primary_key,
            -- Foreign Key Info
            CASE 
                WHEN fk.column_name IS NOT NULL THEN true 
                ELSE false 
            END as is_foreign_key,
            fk.foreign_table_name,
            fk.foreign_column_name,
            fk.constraint_name as fk_constraint_name,
            -- Check constraints
            cc.check_clause,
            -- Unique constraints
            CASE 
                WHEN uc.column_name IS NOT NULL THEN true 
                ELSE false 
            END as is_unique
        FROM
            information_schema.columns AS c
        -- Primary Keys
        LEFT JOIN (
            SELECT
                kcu.table_name,
                kcu.column_name,
                tc.constraint_name
            FROM
                information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu 
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
            WHERE
                tc.constraint_type = 'PRIMARY KEY'
                AND tc.table_schema = 'public'
        ) pk ON c.table_name = pk.table_name AND c.column_name = pk.column_name
        -- Foreign Keys
        LEFT JOIN (
            SELECT
                kcu.table_name,
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name,
                tc.constraint_name
            FROM
                information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
                    AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_schema = 'public'
        ) fk ON c.table_name = fk.table_name AND c.column_name = fk.column_name
        -- Check Constraints
        LEFT JOIN (
            SELECT
                tc.table_name,
                kcu.column_name,
                cc.check_clause
            FROM
                information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.check_constraints cc
                    ON tc.constraint_name = cc.constraint_name
            WHERE
                tc.constraint_type = 'CHECK'
                AND tc.table_schema = 'public'
        ) cc ON c.table_name = cc.table_name AND c.column_name = cc.column_name
        -- Unique Constraints
        LEFT JOIN (
            SELECT
                kcu.table_name,
                kcu.column_name
            FROM
                information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
            WHERE
                tc.constraint_type = 'UNIQUE'
                AND tc.table_schema = 'public'
        ) uc ON c.table_name = uc.table_name AND c.column_name = uc.column_name
        WHERE
            c.table_schema = 'public'
        ORDER BY
            c.table_name, c.ordinal_position;
        """

        print("📊 Fetching schema information...")
        cur.execute(schema_query)
        rows = cur.fetchall()
        print(f"📋 Found {len(rows)} columns across all tables")

        # Process results
        tables = {}
        for row in rows:
            (table_name, column_name, data_type, is_nullable, column_default,
             char_max_length, numeric_precision, numeric_scale, ordinal_position,
             is_primary_key, is_foreign_key, foreign_table_name, foreign_column_name,
             fk_constraint_name, check_clause, is_unique) = row
            
            # Initialize table if not exists
            if table_name not in tables:
                tables[table_name] = {
                    "table_name": table_name,
                    "columns": [],
                    "primary_keys": [],
                    "foreign_keys": [],
                    "unique_constraints": [],
                    "check_constraints": []
                }
            
            # Build column info
            column_info = {
                "name": column_name,
                "type": data_type,
                "nullable": is_nullable == "YES",
                "default": column_default,
                "position": ordinal_position,
                "constraints": {
                    "primary_key": is_primary_key,
                    "foreign_key": is_foreign_key,
                    "unique": is_unique,
                    "check": check_clause is not None
                }
            }
            
            # Add type-specific details
            if char_max_length:
                column_info["max_length"] = char_max_length
            if numeric_precision:
                column_info["precision"] = numeric_precision
            if numeric_scale:
                column_info["scale"] = numeric_scale
            if check_clause:
                column_info["check_constraint"] = check_clause
            
            # Add foreign key reference
            if is_foreign_key and foreign_table_name:
                column_info["references"] = {
                    "table": foreign_table_name,
                    "column": foreign_column_name,
                    "constraint_name": fk_constraint_name
                }
            
            tables[table_name]["columns"].append(column_info)
            
            # Track constraints at table level
            if is_primary_key and column_name not in tables[table_name]["primary_keys"]:
                tables[table_name]["primary_keys"].append(column_name)
            
            if is_foreign_key and foreign_table_name:
                fk_info = {
                    "column": column_name,
                    "references_table": foreign_table_name,
                    "references_column": foreign_column_name,
                    "constraint_name": fk_constraint_name
                }
                if fk_info not in tables[table_name]["foreign_keys"]:
                    tables[table_name]["foreign_keys"].append(fk_info)
            
            if is_unique and column_name not in tables[table_name]["unique_constraints"]:
                tables[table_name]["unique_constraints"].append(column_name)
            
            if check_clause and check_clause not in tables[table_name]["check_constraints"]:
                tables[table_name]["check_constraints"].append(check_clause)

        # Create final JSON structure
        schema_output = {
            "database_info": {
                "extracted_at": datetime.now().isoformat(),
                "source": "Railway PostgreSQL",
                "total_tables": len(tables)
            },
            "tables": list(tables.values())
        }

        # Save to file
        with open(output_filename, 'w') as f:
            json.dump(schema_output, f, indent=2, default=str)

        print(f"💾 Schema saved to '{output_filename}'")
        print(f"\n📊 Summary:")
        print(f"   Total tables: {len(tables)}")
        
        for table_name, table_info in tables.items():
            pk_count = len(table_info["primary_keys"])
            fk_count = len(table_info["foreign_keys"])
            col_count = len(table_info["columns"])
            print(f"   • {table_name}: {col_count} columns, {pk_count} PKs, {fk_count} FKs")

        return schema_output

    except psycopg2.Error as e:
        print(f"❌ Database connection error: {e}")
        print("💡 Make sure you're using the correct PUBLIC Railway URL")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None
    finally:
        if conn:
            cur.close()
            conn.close()
            print("🔌 Database connection closed")

if __name__ == "__main__":
    load_dotenv()

    # Get database URL from environment variables, prioritizing the public one
    DATABASE_URL = os.getenv('DATABASE_URL_PUBLIC') or os.getenv('DATABASE_URL')

    print("🚂 Railway PostgreSQL Schema Extractor")
    print("=" * 50)

    if not DATABASE_URL:
        print("❌ DATABASE_URL not found in environment variables or .env file.")
        print("   Please create a .env file or set the DATABASE_URL_PUBLIC/DATABASE_URL environment variable.")
    else:
        result = get_db_schema_as_json(DATABASE_URL, 'railway_schema.json')

        if result:
            print("\n✅ Schema extraction completed successfully!")
            print("📁 Check 'railway_schema.json' for the complete schema")
        else:
            print("\n❌ Schema extraction failed")