from heart.data import COLUMNS, clean, load_raw

RAW = (
  "63,1,1,145,233,1,2,150,0,2.3,3,0.0,6.0,0\n"
  "67,1,4,160,286,0,2,108,1,1.5,2,3.0,3.0,2\n"
  "41,0,2,130,204,0,2,172,0,1.4,1,?,3.0,0\n"
)


def test_load_raw_parses_question_marks_as_nan(tmp_path):
  p = tmp_path / "h.data"
  p.write_text(RAW)
  df = load_raw(str(p))
  assert list(df.columns) == COLUMNS
  assert df["ca"].isna().sum() == 1


def test_clean_binarises_target_and_dedups(tmp_path):
  p = tmp_path / "h.data"
  p.write_text(RAW + RAW.splitlines()[0] + "\n")
  out = clean(load_raw(str(p)))
  assert set(out["target"].unique()) == {0, 1}
  assert len(out) == 3
  assert "num" not in out.columns


def test_download_bad_zip_raises_and_leaves_no_file(tmp_path, monkeypatch):
  import io
  import urllib.request

  import pytest

  from heart.data import download

  class Resp(io.BytesIO):
    def __enter__(self):
      return self

    def __exit__(self, *a):
      return False

  monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: Resp(b"not a zip"))
  dest = tmp_path / "raw" / "x.data"
  with pytest.raises(RuntimeError, match="download"):
    download(str(dest))
  assert not dest.exists()
