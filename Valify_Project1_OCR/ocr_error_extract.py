from datetime import datetime, timedelta, timezone  # datetime/timedelta for dates, timezone for UTC

from airflow.sdk import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.timetables.trigger import CronTriggerTimetable  # runs the DAG at a fixed clock time (4:00 AM) #####

# Apache Airflow orchestrates the data pipeline (when to run, how to handle failures, backfills etc.),
# and PostgresHook is used to connect to the Postgres database.

SOURCE_CONN_ID = "core_postgres"      # Core's database (OCR_data_full): we only READ from it
TARGET_CONN_ID = "pipeline_postgres"  # our own database (ocr_pipeline): the 4201 rows are WRITTEN here
TARGET_TABLE = "ocr_error_4201"       # table inside ocr_pipeline that stores the extracted rows

# Target table. Created on the first run, skipped on every run after that.
CREATE_TABLE = f"""
    CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
        id              integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,  -- our own row number: 1, 2, 3... in load order
        transaction_id  integer NOT NULL UNIQUE,             -- transaction id from Core; UNIQUE blocks duplicates
        "time"          timestamptz NOT NULL,                -- when the transaction happened (from Core)
        error_codes     text NOT NULL,                       -- just the code, e.g. 4201 (the message text after it is dropped)
        api_version     text,                                -- API version the transaction used (from Core)
        ingested_at     timestamptz NOT NULL DEFAULT now()   -- when this pipeline loaded the row
    )
"""

LAST_ID_QUERY = f"SELECT COALESCE(MAX(transaction_id), 0) FROM {TARGET_TABLE}"  # checkpoint: highest transaction id already loaded (0 when empty)
SOURCE_MAX_ID_QUERY = "SELECT COALESCE(MAX(id), 0) FROM services_transaction"  # newest id in Core at the start of the run

# Which rows to copy from Core. The % is doubled because the query is sent with parameters.
SOURCE_FILTER = """
    FROM services_transaction
    WHERE error_codes LIKE '4201%%'                                                -- error code starts with 4201
      AND id >  %(after_id)s                                                       -- only records after the checkpoint
      AND id <= %(up_to_id)s                                                       -- ...up to the newest id seen when the run started
      AND (%(start)s::timestamptz IS NULL OR "time" >= %(start)s::timestamptz)    -- optional -c range: first day (skipped when NULL)
      AND (%(end)s::timestamptz   IS NULL OR "time" <  %(end)s::timestamptz)      -- optional -c range: day after the last day
"""
SOURCE_QUERY = (
    'SELECT id, "time", left(error_codes, 4), api_version'  # Core's id becomes our transaction_id; left(..., 4) keeps only "4201"
    + SOURCE_FILTER
    + "ORDER BY id"                                          # oldest transaction first, so our row numbers follow Core's order
)

INSERT_ROW = f"""
    INSERT INTO {TARGET_TABLE} (transaction_id, "time", error_codes, api_version)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (transaction_id) DO NOTHING
"""  # id is not listed: Postgres fills it in (1, 2, 3...). ON CONFLICT DO NOTHING: never overwrite a loaded row
EXISTING_IDS_QUERY = f"SELECT transaction_id FROM {TARGET_TABLE} WHERE transaction_id = ANY(%s)"  # which of the given transactions are already in our table


@dag(
    schedule=CronTriggerTimetable("0 4 * * *", timezone="Africa/Cairo"),  # every day at 04:00 Cairo time #####
    start_date=datetime(2026, 9, 28),   # the schedule is active from this date. 
    catchup=False,                     # no run per past day: the checkpoint already makes the first run load everything #####
    default_args={"retries": 3, "retry_delay": timedelta(minutes=5)},
)
def ocr_error_extract():

    @task
    def prepare(dag_run=None):  # decides which records this run covers
        target = PostgresHook(postgres_conn_id=TARGET_CONN_ID)  # connection to our database
        source = PostgresHook(postgres_conn_id=SOURCE_CONN_ID)  # connection to Core's database
        target.run(CREATE_TABLE)                                 # make sure the target table exists
        (up_to_id,) = source.get_first(SOURCE_MAX_ID_QUERY)      # snapshot: newest id in Core right now
        conf = dag_run.conf or {}                                # dates passed with -c (empty on scheduled runs)
        if not conf:                                             # normal run: continue from the checkpoint
            (after_id,) = target.get_first(LAST_ID_QUERY)        # highest transaction id we already loaded
            window = {"after_id": after_id, "up_to_id": up_to_id, "start": None, "end": None}  # no date limits
        else:                                                    # -c run: reload a date range
            if "start" not in conf or "end" not in conf:         # both dates are required
                raise ValueError('Pass both dates, e.g. -c \'{"start": "2025-01-01", "end": "2025-12-31"}\'')
            first_day = datetime.fromisoformat(conf["start"]).replace(tzinfo=timezone.utc)  # start of the first day (UTC)
            last_day = datetime.fromisoformat(conf["end"]).replace(tzinfo=timezone.utc)     # start of the last day (UTC)
            window = {
                "after_id": 0,                                              # ignore the checkpoint: look at every id
                "up_to_id": up_to_id,                                       # still stop at the snapshot
                "start": first_day.isoformat(),                             # first day included
                "end": (last_day + timedelta(days=1)).isoformat(),          # last day included (stop at the next midnight)
            }
        print(f"Window for this run: {window}")                  # shows up in the log
        return window                                            # handed to the next tasks

    @task
    def load(window):  # copies the rows from Core into our table
        source = PostgresHook(postgres_conn_id=SOURCE_CONN_ID)  
        target = PostgresHook(postgres_conn_id=TARGET_CONN_ID)  
        rows = source.get_records(SOURCE_QUERY, parameters=window)  # the 4201 rows in this window
        if not rows:                                             # nothing new since the last run
            print("Extracted 0 rows, nothing to load")
            return {"extracted": 0, "in_target": 0}
        transaction_ids = [row[0] for row in rows]               # transaction ids of the extracted rows
        existing = {r[0] for r in target.get_records(EXISTING_IDS_QUERY, parameters=(transaction_ids,))}  # already loaded earlier
        new_rows = [row for row in rows if row[0] not in existing]  # only insert new ones, so the row numbers have no gaps
        if new_rows:                                             # skip the insert when everything is already loaded
            conn = target.get_conn()                             # one connection for all inserts
            with conn.cursor() as cur:
                cur.executemany(INSERT_ROW, new_rows)            # insert the new rows (Postgres numbers them)
            conn.commit()                                        # save the inserts
            conn.close()
        in_target = len(target.get_records(EXISTING_IDS_QUERY, parameters=(transaction_ids,)))  # how many are in our table now
        print(f"Extracted {len(rows)} rows: {len(new_rows)} new, {len(existing)} already loaded")
        return {"extracted": len(rows), "in_target": in_target}

    @task
    def reconcile(loaded):  #checks every extracted row actually reached the target table
        if loaded["in_target"] != loaded["extracted"]:           # some rows did not reach the target table
            raise ValueError(f"Only {loaded['in_target']} of {loaded['extracted']} extracted rows are in {TARGET_TABLE}")
        print(f"Reconciled: {loaded['extracted']} rows extracted, all present in {TARGET_TABLE}")  # all rows accounted for

    window = prepare()                                          
    reconcile(load(window))                                      


ocr_error_extract()
