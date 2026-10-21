"""
Unit tests for the data_store package.

Run with:
    python -m unittest data_store.tests.test_data_store -v
"""

import unittest

from data_store import FaceRegistryDB, CorrectionLogDB, DataStoreError


class TestFaceRegistryDB(unittest.TestCase):
    def setUp(self):
        self.db = FaceRegistryDB(":memory:")  # fresh in-memory DB per test

    def tearDown(self):
        self.db.close()

    def test_enroll_and_retrieve(self):
        row_id = self.db.enroll("Alice", [0.1, 0.2, 0.3], source="photo1.jpg")
        self.assertIsInstance(row_id, int)
        embeddings = self.db.embeddings_for("Alice")
        self.assertEqual(len(embeddings), 1)
        self.assertEqual(embeddings[0].embedding, [0.1, 0.2, 0.3])
        self.assertEqual(embeddings[0].source, "photo1.jpg")

    def test_multiple_exemplars_per_person(self):
        self.db.enroll("Alice", [0.1, 0.2], source="a")
        self.db.enroll("Alice", [0.3, 0.4], source="b")
        self.assertEqual(len(self.db.embeddings_for("Alice")), 2)

    def test_list_people_distinct(self):
        self.db.enroll("Alice", [0.1], source="a")
        self.db.enroll("Alice", [0.2], source="b")
        self.db.enroll("Bob", [0.3], source="c")
        self.assertEqual(self.db.list_people(), ["Alice", "Bob"])

    def test_all_embeddings_across_everyone(self):
        self.db.enroll("Alice", [0.1])
        self.db.enroll("Bob", [0.2])
        self.assertEqual(len(self.db.all_embeddings()), 2)

    def test_remove_person(self):
        self.db.enroll("Alice", [0.1])
        self.db.enroll("Alice", [0.2])
        removed = self.db.remove_person("Alice")
        self.assertEqual(removed, 2)
        self.assertEqual(self.db.embeddings_for("Alice"), [])

    def test_empty_person_name_raises(self):
        with self.assertRaises(DataStoreError):
            self.db.enroll("", [0.1, 0.2])

    def test_empty_embedding_raises(self):
        with self.assertRaises(DataStoreError):
            self.db.enroll("Alice", [])

    def test_count(self):
        self.db.enroll("Alice", [0.1])
        self.db.enroll("Bob", [0.2])
        self.assertEqual(self.db.count(), 2)

    def test_persists_across_reconnect(self):
        import tempfile, os
        tmp_path = tempfile.mktemp(suffix=".db")
        try:
            db1 = FaceRegistryDB(tmp_path)
            db1.enroll("Alice", [0.1, 0.2, 0.3])
            db1.close()

            db2 = FaceRegistryDB(tmp_path)  # reopen same file
            self.assertEqual(db2.embeddings_for("Alice")[0].embedding, [0.1, 0.2, 0.3])
            db2.close()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestCorrectionLogDB(unittest.TestCase):
    def setUp(self):
        self.db = CorrectionLogDB(":memory:")

    def tearDown(self):
        self.db.close()

    def test_record_and_query(self):
        row_id = self.db.record_decision(
            frame_id="clip1_042", bbox=[10, 20, 100, 200], decision="accepted",
            detected_class="credit_card", confidence=0.81, reviewer="alice",
        )
        self.assertIsInstance(row_id, int)

        records = self.db.get_decisions(frame_id="clip1_042")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].decision, "accepted")
        self.assertEqual(records[0].bbox, [10, 20, 100, 200])
        self.assertEqual(records[0].reviewer, "alice")

    def test_invalid_decision_raises(self):
        with self.assertRaises(DataStoreError):
            self.db.record_decision(frame_id="f1", bbox=[0, 0, 1, 1], decision="maybe")

    def test_bad_bbox_length_raises(self):
        with self.assertRaises(DataStoreError):
            self.db.record_decision(frame_id="f1", bbox=[0, 0, 1], decision="accepted")

    def test_filter_by_decision(self):
        self.db.record_decision(frame_id="f1", bbox=[0, 0, 1, 1], decision="accepted")
        self.db.record_decision(frame_id="f2", bbox=[0, 0, 1, 1], decision="rejected")
        self.db.record_decision(frame_id="f3", bbox=[0, 0, 1, 1], decision="accepted")

        accepted = self.db.get_decisions(decision="accepted")
        self.assertEqual(len(accepted), 2)
        self.assertEqual(self.db.count(decision="rejected"), 1)

    def test_limit_respected(self):
        for i in range(10):
            self.db.record_decision(frame_id=f"f{i}", bbox=[0, 0, 1, 1], decision="accepted")
        self.assertEqual(len(self.db.get_decisions(limit=3)), 3)

    def test_most_recent_first(self):
        self.db.record_decision(frame_id="first", bbox=[0, 0, 1, 1], decision="accepted")
        self.db.record_decision(frame_id="second", bbox=[0, 0, 1, 1], decision="accepted")
        records = self.db.get_decisions()
        self.assertEqual(records[0].frame_id, "second")  # newest first


class TestAPI(unittest.TestCase):
    def setUp(self):
        # Each test gets its own isolated in-memory-backed app instance by
        # monkeypatching the module-level DB handles before importing the app.
        import data_store.api as api_module
        from fastapi.testclient import TestClient

        api_module._face_db = FaceRegistryDB(":memory:")
        api_module._correction_db = CorrectionLogDB(":memory:")
        self.client = TestClient(api_module.app)

    def test_enroll_face_endpoint(self):
        response = self.client.post("/faces", json={"person_name": "Alice", "embedding": [0.1, 0.2, 0.3]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["person_name"], "Alice")

    def test_list_faces_endpoint(self):
        self.client.post("/faces", json={"person_name": "Alice", "embedding": [0.1]})
        response = self.client.get("/faces")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Alice", response.json()["people"])

    def test_enroll_face_bad_request(self):
        response = self.client.post("/faces", json={"person_name": "", "embedding": [0.1]})
        self.assertEqual(response.status_code, 400)

    def test_record_and_get_correction_endpoints(self):
        post_response = self.client.post("/corrections", json={
            "frame_id": "clip1_042", "bbox": [10, 20, 100, 200], "decision": "accepted",
            "detected_class": "credit_card", "confidence": 0.81, "reviewer": "alice",
        })
        self.assertEqual(post_response.status_code, 200)

        get_response = self.client.get("/corrections", params={"frame_id": "clip1_042"})
        self.assertEqual(get_response.status_code, 200)
        body = get_response.json()
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["corrections"][0]["decision"], "accepted")

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
