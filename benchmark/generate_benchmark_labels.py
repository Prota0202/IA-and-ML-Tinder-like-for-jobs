import json
import csv

with open("benchmark/test_cvs.json") as f:
    cvs = [cv["id"] for cv in json.load(f)]
with open("benchmark/test_jobs.json") as f:
    jobs = [job["id"] for job in json.load(f)]

rows = []
for cv in cvs:
    for job in jobs:
        rows.append({"id_cv": cv, "id_job": job, "ground_truth": ""})  # à renseigner à la main

with open("benchmark/labels.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["id_cv", "id_job", "ground_truth"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Benchmark: {len(rows)} paires générées dans benchmark/labels.csv. À annoter !")