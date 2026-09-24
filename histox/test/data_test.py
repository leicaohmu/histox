import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from histox import data


class _FakeResponse:

    def __init__(self, content, status_code=200, headers=None):
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}
        self.closed = False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        del chunk_size
        yield self.content

    def close(self):
        self.closed = True


class DatasetRegistryTest(unittest.TestCase):

    def test_builtin_metadata_fixture(self):
        records = data.list_datasets()

        self.assertIn("histox-metadata-fixture", [record.name for record in records])
        record = data.get_dataset("histox-metadata-fixture")
        self.assertEqual(record.version, "1.0")
        self.assertTrue(record.metadata_only)
        self.assertEqual(record.provider.name, "HistoX")

        download_record = data.get_dataset("histox-download-fixture")
        self.assertFalse(download_record.metadata_only)
        self.assertEqual(download_record.assets[0].checksum_algorithm, "sha256")

    def test_unknown_dataset_and_version(self):
        with self.assertRaises(data.DatasetNotFoundError):
            data.get_dataset("not-a-dataset")
        with self.assertRaises(data.DatasetNotFoundError):
            data.get_dataset("histox-metadata-fixture", version="missing")

    def test_cache_resolution_precedence(self):
        with mock.patch.dict(
            os.environ,
            {
                "HISTOX_CACHE_DIR": "/shared/histox-cache",
                "XDG_CACHE_HOME": "/ignored/xdg-cache",
            },
            clear=True,
        ):
            self.assertEqual(
                data.resolve_cache_root(), Path("/shared/histox-cache")
            )
            self.assertEqual(
                data.resolve_cache_root("/explicit/cache"),
                Path("/explicit/cache"),
            )

        with mock.patch.dict(
            os.environ,
            {"XDG_CACHE_HOME": "/xdg-cache"},
            clear=True,
        ):
            self.assertEqual(
                data.resolve_cache_root(), Path("/xdg-cache/histox")
            )

    def test_metadata_plan_is_read_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = Path(temp_dir) / "cache"
            plan = data.plan_download("histox-metadata-fixture", path=cache)

            self.assertEqual(
                plan.destination,
                cache / "histox-metadata-fixture" / "1.0",
            )
            self.assertEqual(plan.items, ())
            self.assertEqual(plan.pending_count, 0)
            self.assertFalse(plan.destination.exists())


class DownloadPlanTest(unittest.TestCase):

    @staticmethod
    def _record():
        return data.DatasetRecord(
            name="test-dataset",
            version="1",
            title="Test dataset",
            description="A local record used for unit testing.",
            provider=data.ProviderRecord(
                name="Test provider",
                dataset_id="test:1",
                url="https://example.org/datasets/test",
            ),
            access="open",
            license="CC0-1.0",
            terms_url="https://example.org/terms",
            assets=(
                data.AssetRecord(
                    path="slides/example.svs",
                    url="https://example.org/example.svs",
                    size_bytes=4,
                    checksum="098f6bcd4621d373cade4e832627b4f6",
                    checksum_algorithm="md5",
                ),
                data.AssetRecord(
                    path="annotations.csv",
                    url="https://example.org/annotations.csv",
                ),
            ),
        )

    def test_missing_present_mismatch_and_conflict_states(self):
        record = self._record()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = data.plan_download(record, path=root)
            self.assertEqual(
                [item.status for item in first.items],
                ["missing", "missing"],
            )
            self.assertEqual(first.total_size_bytes, 4)
            self.assertEqual(first.download_size_bytes, 4)
            self.assertEqual(first.unknown_size_count, 1)

            slide = first.items[0].destination
            slide.parent.mkdir(parents=True)
            slide.write_bytes(b"test")
            second = data.plan_download(record, path=root)
            self.assertEqual(
                [item.status for item in second.items],
                ["present", "missing"],
            )

            slide.write_bytes(b"no")
            annotation = second.items[1].destination
            annotation.mkdir()
            third = data.plan_download(record, path=root)
            self.assertEqual(
                [item.status for item in third.items],
                ["size-mismatch", "conflict"],
            )
            self.assertEqual(third.pending_count, 2)
            serialized = third.to_dict()
            self.assertEqual(serialized["schema_version"], 1)
            self.assertEqual(serialized["access"], "open")
            self.assertEqual(serialized["license"], "CC0-1.0")

    def test_records_reject_unsafe_or_incomplete_assets(self):
        with self.assertRaises(ValueError):
            data.AssetRecord(path="../outside.svs")
        with self.assertRaises(ValueError):
            data.AssetRecord(path=".")
        with self.assertRaises(ValueError):
            data.AssetRecord(path="slide.svs", checksum="abc")

    def test_record_and_version_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            data.plan_download(self._record(), version="1")


class DatasetDownloadTest(unittest.TestCase):

    @staticmethod
    def _record(access="open", checksum=None):
        content = b"test"
        return data.DatasetRecord(
            name="download-fixture",
            version="1",
            title="Download fixture",
            description="A direct HTTP fixture used with mocked responses.",
            provider=data.ProviderRecord(
                name="Test provider",
                dataset_id="test:download",
                url="https://example.org/datasets/test",
            ),
            access=access,
            license="CC0-1.0",
            terms_url="https://example.org/terms",
            assets=(
                data.AssetRecord(
                    path="slides/example.svs",
                    url="https://example.org/example.svs",
                    size_bytes=len(content),
                    checksum=checksum or hashlib.sha256(content).hexdigest(),
                    checksum_algorithm="sha256",
                ),
            ),
        )

    def test_download_is_atomic_and_writes_provenance(self):
        response = _FakeResponse(b"test")
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch(
            "histox.data._download.requests.get", return_value=response
        ) as request:
            result = data.download_dataset(self._record(), path=temp_dir)

            target = result.destination / "slides" / "example.svs"
            self.assertEqual(target.read_bytes(), b"test")
            self.assertFalse(target.with_name(target.name + ".part").exists())
            self.assertEqual(result.downloaded_count, 1)
            self.assertEqual(result.reused_count, 0)
            self.assertTrue(response.closed)
            request.assert_called_once()

            provenance = json.loads(result.provenance_path.read_text())
            self.assertEqual(provenance["schema_version"], 1)
            self.assertEqual(provenance["dataset"], "download-fixture")
            self.assertEqual(provenance["assets"][0]["action"], "downloaded")
            self.assertEqual(provenance["assets"][0]["validation"], "checksum")

    def test_valid_existing_file_is_reused_without_network(self):
        record = self._record()
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = data.plan_download(record, path=temp_dir)
            target = plan.items[0].destination
            target.parent.mkdir(parents=True)
            target.write_bytes(b"test")

            with mock.patch("histox.data._download.requests.get") as request:
                result = data.download_dataset(record, path=temp_dir)

            request.assert_not_called()
            self.assertEqual(result.downloaded_count, 0)
            self.assertEqual(result.reused_files, (target,))

    def test_partial_file_resumes_with_range_request(self):
        record = self._record()
        response = _FakeResponse(
            b"st",
            status_code=206,
            headers={"Content-Range": "bytes 2-3/4"},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            target = data.plan_download(record, path=temp_dir).items[0].destination
            target.parent.mkdir(parents=True)
            target.with_name(target.name + ".part").write_bytes(b"te")

            with mock.patch(
                "histox.data._download.requests.get", return_value=response
            ) as request:
                result = data.download_dataset(record, path=temp_dir)

            self.assertEqual(target.read_bytes(), b"test")
            self.assertEqual(result.resumed_files, (target,))
            self.assertEqual(
                request.call_args.kwargs["headers"], {"Range": "bytes=2-"}
            )

    def test_provider_without_range_support_restarts_partial_file(self):
        record = self._record()
        response = _FakeResponse(b"test", status_code=200)
        with tempfile.TemporaryDirectory() as temp_dir:
            target = data.plan_download(record, path=temp_dir).items[0].destination
            target.parent.mkdir(parents=True)
            target.with_name(target.name + ".part").write_bytes(b"te")

            with mock.patch(
                "histox.data._download.requests.get", return_value=response
            ):
                result = data.download_dataset(record, path=temp_dir)

            self.assertEqual(target.read_bytes(), b"test")
            self.assertEqual(result.resumed_count, 0)

    def test_integrity_failure_does_not_publish_partial_asset(self):
        response = _FakeResponse(b"nope")
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch(
            "histox.data._download.requests.get", return_value=response
        ):
            plan = data.plan_download(self._record(), path=temp_dir)
            target = plan.items[0].destination
            with self.assertRaises(data.DatasetIntegrityError):
                data.download_dataset(self._record(), path=temp_dir)

            self.assertFalse(target.exists())
            self.assertFalse(target.with_name(target.name + ".part").exists())
            self.assertFalse((plan.destination / ".histox-provenance.json").exists())

    def test_existing_invalid_asset_requires_explicit_overwrite(self):
        record = self._record()
        with tempfile.TemporaryDirectory() as temp_dir:
            target = data.plan_download(record, path=temp_dir).items[0].destination
            target.parent.mkdir(parents=True)
            target.write_bytes(b"nope")

            with mock.patch("histox.data._download.requests.get") as request:
                with self.assertRaises(data.DatasetConflictError):
                    data.download_dataset(record, path=temp_dir)

            request.assert_not_called()
            self.assertEqual(target.read_bytes(), b"nope")

    def test_controlled_access_is_not_sent_to_generic_downloader(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch(
            "histox.data._download.requests.get"
        ) as request:
            with self.assertRaises(data.DatasetAccessError):
                data.download_dataset(
                    self._record(access="controlled"), path=temp_dir
                )

            request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
