"""Automatic Acceptance Tests for Phase E (Drop-your-data feature).

Starts uvicorn on port 8000 using .venv311, waits for health, runs checks A-G,
then stops uvicorn and confirms port 8000 is free.
"""

import io
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import requests

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
API_URL = "http://127.0.0.1:8000"


def is_port_open(port: int = 8000) -> bool:
    """Return True if port is accepting TCP connections."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait_for_server(url: str, timeout: float = 30.0) -> bool:
    """Poll limits endpoint until server is healthy."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{url}/dropbox/limits", timeout=1.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def kill_server(proc: subprocess.Popen, port: int = 8000) -> None:
    """Terminate uvicorn process tree and wait until port is free."""
    print("\nStopping uvicorn server...")
    try:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            text=True,
        )
    except Exception:
        proc.kill()

    # Wait for port to be free
    start = time.time()
    while time.time() - start < 10.0:
        if not is_port_open(port):
            print(f"Confirmed: Port {port} is free.")
            return
        time.sleep(0.5)

    if not is_port_open(port):
        print(f"Confirmed: Port {port} is free.")
    else:
        print(f"Warning: Port {port} still appears open.")


def main():
    print("=" * 60)
    print("PHASE E: AUTOMATIC ACCEPTANCE TESTS (scripts/acceptance_dropbox.py)")
    print("=" * 60)

    # 1. Start uvicorn server
    python_exe = sys.executable
    print(f"Python executable: {python_exe}")
    print("Starting uvicorn server in background on 127.0.0.1:8000...")
    
    server_proc = subprocess.Popen(
        [python_exe, "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    results = {}
    check_b_data_300 = None

    try:
        print("Waiting for server health check on /dropbox/limits...")
        healthy = wait_for_server(API_URL, timeout=30.0)
        if not healthy:
            raise RuntimeError("Server failed to start and respond on /dropbox/limits within 30 seconds.")
        print("Server is up and healthy!\n")

        # -------------------------------------------------------------
        # STEP 2A & 2B: Cold start & Held-out accuracy
        # -------------------------------------------------------------
        print("--- Step 2A: Cold start measurement ---")
        test_parquet_path = os.path.join(REPO_ROOT, "data", "processed", "test.parquet")
        test_df = pd.read_parquet(test_parquet_path)

        # Sample 300 rows (random_state=3)
        sample_300 = test_df.sample(300, random_state=3).copy()
        sample_300["stars"] = sample_300["sentiment"].map({"positive": 5, "negative": 1})
        sample_300["review"] = sample_300["text"]
        csv_300_bytes = sample_300[["review", "stars"]].to_csv(index=False).encode("utf-8")

        temp_dir = tempfile.gettempdir()
        temp_csv_300_path = os.path.join(temp_dir, f"test_300_{int(time.time())}.csv")
        try:
            with open(temp_csv_300_path, "wb") as f:
                f.write(csv_300_bytes)

            # First POST: Cold start
            t0 = time.time()
            with open(temp_csv_300_path, "rb") as f:
                resp_cold = requests.post(
                    f"{API_URL}/dropbox/analyze",
                    files={"file": ("heldout_300.csv", f, "text/csv")},
                    timeout=120.0,
                )
            cold_time = time.time() - t0
            print(f"Cold start (first POST /dropbox/analyze): {cold_time:.2f} s")
            assert resp_cold.status_code == 200, f"Cold start POST failed with {resp_cold.status_code}: {resp_cold.text}"

            # Second POST: Warm start
            t1 = time.time()
            with open(temp_csv_300_path, "rb") as f:
                resp_warm = requests.post(
                    f"{API_URL}/dropbox/analyze",
                    files={"file": ("heldout_300.csv", f, "text/csv")},
                    timeout=120.0,
                )
            warm_time = time.time() - t1
            print(f"Warm start (second POST /dropbox/analyze): {warm_time:.2f} s")
            assert resp_warm.status_code == 200, f"Warm start POST failed with {resp_warm.status_code}"
            
            results["2A_cold_start_time"] = cold_time
            results["2A_warm_start_time"] = warm_time

            # Step 2B Checks on 300 rows
            print("\n--- Step 2B: Held-out accuracy check (300 rows & 1,000 rows) ---")
            data_300 = resp_cold.json()
            check_b_data_300 = data_300
            
            rows_used_300 = data_300["report"]["rows_used"]
            acc_info_300 = data_300.get("accuracy_vs_ratings")
            acc_300 = acc_info_300["accuracy"] if acc_info_300 else 0.0
            macro_f1_300 = acc_info_300["macro_f1"] if acc_info_300 else 0.0
            
            assigned_rows = sum(1 for r in data_300["rows"] if r["theme"] != "Unassigned")
            theme_share_300 = assigned_rows / len(data_300["rows"]) if data_300["rows"] else 0.0
            
            print(f"300 rows - Status: {resp_cold.status_code}")
            print(f"300 rows - rows_used: {rows_used_300} (required >= 290)")
            print(f"300 rows - accuracy: {acc_300:.4f} (required 0.86 - 0.94)")
            print(f"300 rows - macro_f1: {macro_f1_300:.4f}")
            print(f"300 rows - theme assigned share: {theme_share_300:.1%} ({assigned_rows}/{len(data_300['rows'])})")

            pass_b_300 = (
                resp_cold.status_code == 200
                and rows_used_300 >= 290
                and 0.86 <= acc_300 <= 0.94
            )
            print(f"300 rows check PASS: {pass_b_300}")

            # 1,000 rows check (random_state=4)
            sample_1000 = test_df.sample(1000, random_state=4).copy()
            sample_1000["stars"] = sample_1000["sentiment"].map({"positive": 5, "negative": 1})
            sample_1000["review"] = sample_1000["text"]
            csv_1000_bytes = sample_1000[["review", "stars"]].to_csv(index=False).encode("utf-8")
            
            temp_csv_1000_path = os.path.join(temp_dir, f"test_1000_{int(time.time())}.csv")
            try:
                with open(temp_csv_1000_path, "wb") as f:
                    f.write(csv_1000_bytes)
                with open(temp_csv_1000_path, "rb") as f:
                    resp_1000 = requests.post(
                        f"{API_URL}/dropbox/analyze",
                        files={"file": ("heldout_1000.csv", f, "text/csv")},
                        timeout=180.0,
                    )
                assert resp_1000.status_code == 200, f"1000-row POST failed with {resp_1000.status_code}"
                data_1000 = resp_1000.json()
                acc_info_1000 = data_1000.get("accuracy_vs_ratings")
                acc_1000 = acc_info_1000["accuracy"] if acc_info_1000 else 0.0
                macro_f1_1000 = acc_info_1000["macro_f1"] if acc_info_1000 else 0.0
                print(f"1,000 rows - rows_used: {data_1000['report']['rows_used']}")
                print(f"1,000 rows - accuracy: {acc_1000:.4f}")
                print(f"1,000 rows - macro_f1: {macro_f1_1000:.4f}")
            finally:
                if os.path.exists(temp_csv_1000_path):
                    os.remove(temp_csv_1000_path)

            results["2B_pass"] = pass_b_300
            results["2B_300_acc"] = acc_300
            results["2B_300_f1"] = macro_f1_300
            results["2B_300_theme_share"] = theme_share_300
            results["2B_1000_acc"] = acc_1000
            results["2B_1000_f1"] = macro_f1_1000

        finally:
            if os.path.exists(temp_csv_300_path):
                os.remove(temp_csv_300_path)

        # -------------------------------------------------------------
        # STEP 2C: Not-Amazon text
        # -------------------------------------------------------------
        print("\n--- Step 2C: Not-Amazon text check (40 non-product texts) ---")
        non_amazon_texts = [
            # Tweets (10)
            "traffic on the interstate is completely standstill today, absolutely furious with road work",
            "loving the crisp autumn morning air and gorgeous colorful trees today!",
            "customer support at my local electric utility resolved my billing question in five minutes",
            "the new commuter train schedule makes my morning commute twice as long and stressful",
            "congratulations to our neighborhood soccer team on advancing to the state championship!",
            "airline lost my suitcase again and refuses to answer calls or update status",
            "the friendly barista at the corner café gave me a free cinnamon roll this morning",
            "incessant construction noise on Elm Street starting at six am is completely unbearable",
            "just reviewed the quarterly inflation numbers, very alarming data for household expenses",
            "spectacular sunset view over the harbor water this evening, truly peaceful",
            # Support messages (10)
            "can somebody please help reset my credentials for the corporate staff portal?",
            "my checking statement indicates an erroneous duplicate transaction from yesterday",
            "thank you very much for the prompt and courteous help resolving my account inquiry",
            "the online payment portal repeatedly times out whenever I attempt to submit form",
            "I need to update our mailing address before the next billing cycle is processed",
            "the representative was very rude and abruptly disconnected the line when questioned",
            "your technical specialist patiently walked me through the restore steps, truly grateful",
            "where can I download the annual tax documentation statement on the updated portal?",
            "system outage during critical trading hours is completely unacceptable for corporate users",
            "received a swift, thorough, and highly professional resolution from your engineering team",
            # News comments (10)
            "the proposed municipal property tax increase will place heavy burden on retired seniors",
            "thrilled to learn that the municipal district is expanding green space downtown",
            "cutting public transit routes will severely harm working families without automobiles",
            "deeply heartwarming story about community members volunteering to rebuild after storm damage",
            "this zoning modification completely ignores resident input and lacks accountability",
            "breakthrough research in battery chemistry demonstrates promising path toward clean energy",
            "massive potholes along Main Avenue caused severe alignment damage to our vehicle",
            "extending public library weekend operational hours is a tremendous victory for students",
            "cynical politicians making grand promises during campaigns and breaking them immediately after",
            "charity fundraiser exceeded all expectations thanks to generous community participation",
            # General comments (10)
            "our flight departed four hours late without any communication or vouchers provided",
            "had a delightful reunion dinner with college friends at the vintage bistro downtown",
            "meteorologist warned of severe storms but we enjoyed glorious sunshine all afternoon",
            "the dental checkup went smoothly and the hygienist was remarkably gentle and caring",
            "mobile banking application has crashed repeatedly on payroll day when transfers are urgent",
            "the exhibition of antique cartography at the cultural museum was thoroughly educational",
            "hotel ventilation unit sounded like a freight engine and prevented any rest all night",
            "neighborhood cleanup initiative removed over three tons of debris from the creek bank",
            "ruptured water pipe inundated the basement and city maintenance took four hours to respond",
            "attended a delightful acoustic concert in the amphitheater under the stars last weekend"
        ]
        assert len(non_amazon_texts) == 40, f"Expected 40 texts, got {len(non_amazon_texts)}"
        
        df_non_amazon = pd.DataFrame({"text": non_amazon_texts})
        temp_non_amazon_path = os.path.join(temp_dir, f"test_non_amazon_{int(time.time())}.csv")
        try:
            df_non_amazon.to_csv(temp_non_amazon_path, index=False)
            with open(temp_non_amazon_path, "rb") as f:
                resp_c = requests.post(
                    f"{API_URL}/dropbox/analyze",
                    files={"file": ("non_amazon.csv", f, "text/csv")},
                    timeout=60.0,
                )
            assert resp_c.status_code == 200, f"Check C failed with HTTP {resp_c.status_code}: {resp_c.text}"
            data_c = resp_c.json()

            # Check both warnings
            warnings_c = data_c.get("warnings", [])
            w1 = "Models were trained on Amazon product reviews; accuracy on other text may be lower."
            w2 = "Themes are the 29 Amazon-product themes found in this project; they may not fit your data."
            warnings_present = (w1 in warnings_c) and (w2 in warnings_c)
            print(f"HTTP Status: {resp_c.status_code}")
            print(f"Both warnings present: {warnings_present}")
            
            # Sentiment split
            rows_c = data_c.get("rows", [])
            pos_c = sum(1 for r in rows_c if r["label"] == "positive")
            neg_c = sum(1 for r in rows_c if r["label"] == "negative")
            print(f"Sentiment split: {pos_c} positive ({pos_c/len(rows_c)*100:.1f}%), {neg_c} negative ({neg_c/len(rows_c)*100:.1f}%)")

            # Share Unassigned
            unassigned_c = sum(1 for r in rows_c if r["theme"] == "Unassigned")
            share_unassigned_c = unassigned_c / len(rows_c)
            print(f"Share 'Unassigned': {share_unassigned_c:.1%} ({unassigned_c}/{len(rows_c)})")

            # Share prob < 0.65
            low_conf_c = sum(1 for r in rows_c if r["probability"] < 0.65)
            share_low_conf_c = low_conf_c / len(rows_c)
            print(f"Share probability < 0.65: {share_low_conf_c:.1%} ({low_conf_c}/{len(rows_c)})")

            # 5 example rows
            print("\n5 Example Rows from Not-Amazon text:")
            for ex in rows_c[:5]:
                print(f"  Row {ex['row_id']}: [{ex['label'].upper()} {ex['probability']:.4f}] Theme: '{ex['theme']}' | Text: {ex['scrubbed_text'][:65]}...")

            pass_c = resp_c.status_code == 200 and warnings_present
            print(f"\nCheck C PASS: {pass_c}")
            results["2C_pass"] = pass_c
            results["2C_pos"] = pos_c
            results["2C_neg"] = neg_c
            results["2C_unassigned_share"] = share_unassigned_c
            results["2C_low_conf_share"] = share_low_conf_c

        finally:
            if os.path.exists(temp_non_amazon_path):
                os.remove(temp_non_amazon_path)

        # -------------------------------------------------------------
        # STEP 2D: Error handling
        # -------------------------------------------------------------
        print("\n--- Step 2D: Error handling checks ---")
        errors_pass = True

        # 1. .txt file
        resp_txt = requests.post(
            f"{API_URL}/dropbox/analyze",
            files={"file": ("test.txt", io.BytesIO(b"arbitrary text content"), "text/plain")},
        )
        detail_txt = resp_txt.json().get("detail", resp_txt.text)
        print(f"1. .txt file -> Status: {resp_txt.status_code} (expect 415 or 400), Detail: {detail_txt}")
        if resp_txt.status_code not in [400, 415] or "Traceback" in resp_txt.text:
            errors_pass = False

        # 2. Empty CSV
        resp_empty = requests.post(
            f"{API_URL}/dropbox/analyze",
            files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")},
        )
        detail_empty = resp_empty.json().get("detail", resp_empty.text)
        print(f"2. Empty CSV -> Status: {resp_empty.status_code} (expect 400), Detail: {detail_empty}")
        if resp_empty.status_code != 400 or "Traceback" in resp_empty.text:
            errors_pass = False

        # 3. CSV with only numeric columns
        num_csv = "id,value1,value2\n1,100,200\n2,300,400\n".encode("utf-8")
        resp_num = requests.post(
            f"{API_URL}/dropbox/analyze",
            files={"file": ("numeric.csv", io.BytesIO(num_csv), "text/csv")},
        )
        detail_num = resp_num.json().get("detail", resp_num.text)
        print(f"3. Numeric-only CSV -> Status: {resp_num.status_code} (expect 400), Detail: {detail_num}")
        if resp_num.status_code != 400 or "Traceback" in resp_num.text:
            errors_pass = False

        # 4. 6 MB file
        big_content = b"review\n" + b"This is a big review line.\n" * (250_000)  # > 6 MB
        resp_big = requests.post(
            f"{API_URL}/dropbox/analyze",
            files={"file": ("big.csv", io.BytesIO(big_content), "text/csv")},
        )
        detail_big = resp_big.json().get("detail", resp_big.text)
        print(f"4. 6 MB file -> Status: {resp_big.status_code} (expect 413), Detail: {detail_big}")
        if resp_big.status_code != 413 or "Traceback" in resp_big.text:
            errors_pass = False

        # 5. XLSX with 3,500 rows
        print("Creating XLSX with 3,500 rows...")
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["review", "rating"])
        for i in range(3501):
            ws.append([f"Sample feedback text {i}", 5])
        xlsx_buf = io.BytesIO()
        wb.save(xlsx_buf)
        xlsx_buf.seek(0)

        resp_xlsx = requests.post(
            f"{API_URL}/dropbox/analyze",
            files={"file": ("large_sheet.xlsx", xlsx_buf, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )
        detail_xlsx = resp_xlsx.json().get("detail", resp_xlsx.text)
        print(f"5. XLSX with 3,500 rows -> Status: {resp_xlsx.status_code} (expect 400), Detail: {detail_xlsx}")
        if resp_xlsx.status_code != 400 or "Traceback" in resp_xlsx.text:
            errors_pass = False

        print(f"Check D PASS: {errors_pass}")
        results["2D_pass"] = errors_pass
        results["2D_txt_status"] = resp_txt.status_code
        results["2D_empty_status"] = resp_empty.status_code
        results["2D_num_status"] = resp_num.status_code
        results["2D_big_status"] = resp_big.status_code
        results["2D_xlsx_status"] = resp_xlsx.status_code

        # -------------------------------------------------------------
        # STEP 2E: Privacy & Formula guard
        # -------------------------------------------------------------
        print("\n--- Step 2E: Privacy and formula guard check ---")
        privacy_pass = True

        pii_secret_email = "confidential_agent_test987@securemail.org"
        pii_secret_phone = "800-555-0199"
        pii_secret_url = "https://classifiedportal.testnet.internal/auth"

        pii_csv = (
            f"review\n"
            f"Please notify {pii_secret_email} or call {pii_secret_phone} or go to {pii_secret_url} immediately.\n"
            f"=1+1 malicious spreadsheet injection calculation payload\n"
        ).encode("utf-8")

        resp_pii = requests.post(
            f"{API_URL}/dropbox/analyze",
            files={"file": ("privacy_test.csv", io.BytesIO(pii_csv), "text/csv")},
        )
        assert resp_pii.status_code == 200, f"PII call failed with {resp_pii.status_code}"
        rows_pii = resp_pii.json().get("rows", [])
        
        # Check PII scrubbing
        row0_text = rows_pii[0]["scrubbed_text"]
        print(f"Row 0 (Scrubbed PII): {row0_text}")
        if (
            pii_secret_email in row0_text
            or pii_secret_phone in row0_text
            or pii_secret_url in row0_text
            or "<EMAIL>" not in row0_text
            or "<PHONE>" not in row0_text
            or "<URL>" not in row0_text
        ):
            print("ERROR: PII masking failed!")
            privacy_pass = False
        else:
            print("PII Masking: Verified (email, phone, URL replaced with placeholders).")

        # Check Formula guard
        row1_text = rows_pii[1]["scrubbed_text"]
        print(f"Row 1 (Formula Guard): {row1_text}")
        if not row1_text.startswith("'=1+1"):
            print(f"ERROR: Formula guard failed, expected single quote prefix on {row1_text}")
            privacy_pass = False
        else:
            print("Formula Guard: Verified (prefixed with single quote).")

        # Confirm git status shows no new files
        git_status_res = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True)
        untracked = [line for line in git_status_res.stdout.splitlines() if not line.strip().startswith("?? scripts")]
        print(f"Git status modified/untracked files outside scripts/: {untracked}")
        if any(untracked):
            print(f"ERROR: Git status shows unexpected files: {untracked}")
            privacy_pass = False

        # Confirm no uploaded secret appears in any repo log file
        log_files = []
        for root, _, files in os.walk(REPO_ROOT):
            if any(p in root for p in [".git", ".venv", ".venv311", "node_modules"]):
                continue
            for f in files:
                if f.endswith(".log"):
                    log_files.append(os.path.join(root, f))
        
        pii_leaked = False
        for lf in log_files:
            try:
                with open(lf, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    if pii_secret_email in content or pii_secret_phone in content or pii_secret_url in content:
                        print(f"ERROR: PII leaked into log file {lf}!")
                        pii_leaked = True
            except Exception:
                pass
        if pii_leaked:
            privacy_pass = False
        else:
            print("Repo Logs: Confirmed no uploaded PII text appears in any log file in repo.")

        print(f"Check E PASS: {privacy_pass}")
        results["2E_pass"] = privacy_pass

        # -------------------------------------------------------------
        # STEP 2F: Dashboard (streamlit.testing AppTest)
        # -------------------------------------------------------------
        print("\n--- Step 2F: Dashboard AppTest ---")
        from streamlit.testing.v1 import AppTest
        dashboard_file = os.path.join(REPO_ROOT, "app", "dashboard.py")

        # 1. First run: baseline metrics and 0 exceptions
        at = AppTest.from_file(dashboard_file, default_timeout=30)
        at.run()
        print(f"Initial run exceptions: {len(at.exception)}")
        assert len(at.exception) == 0, f"Dashboard initial run raised exceptions: {[e.value for e in at.exception]}"

        metric_vals = [m.value for m in at.metric]
        print(f"Dashboard metrics: {metric_vals}")
        has_50k = "50,000" in metric_vals
        has_pos = "25,331" in metric_vals
        has_neg = "24,669" in metric_vals
        print(f"Overview shows 50,000: {has_50k}, 25,331: {has_pos}, 24,669: {has_neg}")
        assert has_50k and has_pos and has_neg, "Dashboard Overview metrics do not match 50,000 / 25,331 / 24,669"

        # 2. Second run: with st.session_state["dropbox_results"] pre-set to check B JSON
        at2 = AppTest.from_file(dashboard_file, default_timeout=30)
        at2.session_state["dropbox_results"] = check_b_data_300
        at2.run()
        print(f"Run with dropbox_results exceptions: {len(at2.exception)}")
        assert len(at2.exception) == 0, f"Dashboard raised exceptions with dropbox_results: {[e.value for e in at2.exception]}"

        metric_vals_results = [m.value for m in at2.metric]
        print(f"KPI cards rendered: '300' in metrics -> {'300' in metric_vals_results}")
        assert "300" in metric_vals_results, f"Expected 300 rows used in metric values, got {metric_vals_results}"

        # Confirm theme table rendered
        has_theme_markdown = any("Theme Breakdown" in str(m.value) for m in at2.markdown)
        print(f"Theme table section rendered: {has_theme_markdown}")
        assert has_theme_markdown, "Expected Theme Breakdown section in results"

        # Click 'Clear results'
        clear_buttons = [b for b in at2.button if b.key == "dropbox_clear_btn"]
        assert len(clear_buttons) > 0, "Expected 'Clear results' button"
        clear_buttons[0].click().run()
        print(f"Exceptions after click Clear results: {len(at2.exception)}")
        assert len(at2.exception) == 0, f"Exceptions after Clear results: {[e.value for e in at2.exception]}"
        has_key_after_clear = "dropbox_results" in at2.session_state
        print(f"dropbox_results key in session_state after Clear: {has_key_after_clear} (expected False)")
        assert not has_key_after_clear, "dropbox_results key was not removed by Clear results button"

        print("Check F PASS: True")
        results["2F_pass"] = True

        # -------------------------------------------------------------
        # STEP 2G: Snapshot comparison
        # -------------------------------------------------------------
        print("\n--- Step 2G: Snapshot comparison with scratch/api_snapshot.json ---")
        snapshot_file = os.path.join(REPO_ROOT, "scratch", "api_snapshot.json")
        with open(snapshot_file, "r", encoding="utf-8") as f:
            snapshot_data = json.load(f)

        snapshot_eps = ["/sentiment", "/themes", "/summary/4", "/summary/26"]
        snapshot_matches = {}
        for ep in snapshot_eps:
            r = requests.get(f"{API_URL}{ep}", timeout=10.0)
            assert r.status_code == 200, f"Endpoint {ep} returned {r.status_code}"
            live_ep_data = r.json()
            is_match = (live_ep_data == snapshot_data.get(ep))
            snapshot_matches[ep] = is_match
            print(f"{ep}: {is_match}")

        pass_g = all(snapshot_matches.values())
        print(f"Check G PASS: {pass_g}")
        results["2G_pass"] = pass_g
        results["2G_lines"] = snapshot_matches

    finally:
        kill_server(server_proc, port=8000)

    # -------------------------------------------------------------
    # Overall summary and verdict
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE E ACCEPTANCE SUMMARY:")
    print(f"  Step 2A (Cold start): {results.get('2A_cold_start_time', 0):.2f}s, Warm: {results.get('2A_warm_start_time', 0):.2f}s")
    print(f"  Step 2B (Held-out accuracy): PASS = {results.get('2B_pass')}")
    print(f"  Step 2C (Not-Amazon text):  PASS = {results.get('2C_pass')}")
    print(f"  Step 2D (Error handling):   PASS = {results.get('2D_pass')}")
    print(f"  Step 2E (Privacy & Guard):  PASS = {results.get('2E_pass')}")
    print(f"  Step 2F (Dashboard AppTest):PASS = {results.get('2F_pass')}")
    print(f"  Step 2G (Snapshot parity):  PASS = {results.get('2G_pass')}")
    print("=" * 60)

    all_accepted = (
        results.get("2B_pass", False)
        and results.get("2C_pass", False)
        and results.get("2D_pass", False)
        and results.get("2E_pass", False)
        and results.get("2F_pass", False)
        and results.get("2G_pass", False)
    )

    if all_accepted:
        print("ALL ACCEPTANCE CHECKS (B, C, D, E, F, G) PASSED!")
        return 0
    else:
        print("FAIL: One or more acceptance checks failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

