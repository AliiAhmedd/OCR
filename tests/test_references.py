import pytest

from id_classifier.references import storage_key

PREFIXES = ["/mnt/nfs", "media"]


@pytest.mark.parametrize("reference, key", [
    # the three shapes found in services_transactiondatacontainer.meta_data
    ("media/tun_nid_ocr/x.zip", "tun_nid_ocr/x.zip"),
    ("/mnt/nfs/tun_nid_ocr/x.zip", "tun_nid_ocr/x.zip"),
    ("./tun_nid_ocr/x.zip", "tun_nid_ocr/x.zip"),
    # an absolute path under no configured prefix keeps its folders
    ("/srv/data/tun_nid_ocr/x.zip", "srv/data/tun_nid_ocr/x.zip"),
    # untidy spellings of the same key
    ("media\\tun_nid_ocr\\x.zip", "tun_nid_ocr/x.zip"),
    ("  ./media//tun_nid_ocr/./x.zip ", "tun_nid_ocr/x.zip"),
    ("././tun_nid_ocr/x.zip", "tun_nid_ocr/x.zip"),
    # a prefix only matches whole folders
    ("/mnt/nfs2/tun_nid_ocr/x.zip", "mnt/nfs2/tun_nid_ocr/x.zip"),
    ("mediafiles/x.zip", "mediafiles/x.zip"),
    # only the first (leading) folder is stripped, never one further in
    ("tun_nid_ocr/media/x.zip", "tun_nid_ocr/media/x.zip"),
])
def test_references_become_one_key(reference, key):
    assert storage_key(reference, PREFIXES) == key


def test_no_prefixes_only_tidies():
    assert storage_key("/media/a/x.zip") == "media/a/x.zip"


def test_longest_matching_prefix_wins():
    prefixes = ["/app", "/app/media", "./app/media/"]          # written any way: trailing "/" or "./" is fine
    assert storage_key("/app/media/a/x.zip", prefixes) == "a/x.zip"
    assert storage_key("/app/other/x.zip", prefixes) == "other/x.zip"


def test_same_file_gives_same_key_whatever_the_shape():
    shapes = ["media/a/b/x.zip", "/mnt/nfs/a/b/x.zip", "./a/b/x.zip", "a\\b\\x.zip"]
    assert {storage_key(s, PREFIXES) for s in shapes} == {"a/b/x.zip"}


@pytest.mark.parametrize("reference, reason", [
    ("", "empty reference"),
    ("   ", "empty reference"),
    ("https://storage.example.com/a/x.jpg", "reference is a URL"),
    ("s3://bucket/a/x.zip", "reference is a URL"),
    ("media/../secrets/x.zip", "'..'"),
    ("../x.zip", "'..'"),
    ("media/a/", "no file name"),
    ("/", "no file name"),
    ("./", "no file name"),
    ("media/", "no file name"),
])
def test_bad_references_are_refused(reference, reason):
    with pytest.raises(ValueError, match=reason):
        storage_key(reference, PREFIXES)
