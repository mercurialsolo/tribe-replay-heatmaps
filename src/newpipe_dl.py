"""Robust NewPipe+ADB YouTube downloader (survives SABR — NewPipe's extractor works).

Drives the phone via uiautomator (element-based, not fixed coords) to download each video
and pulls it to yt_downloads/<videoID>.mp4. Requires: adb device connected, NewPipe installed,
download folder already granted (we used /sdcard/Videos).

    python3 newpipe_dl.py <id1> <id2> ...
"""
import subprocess as sp
import sys, re, time, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yt_downloads")
VIDEODIR = "/sdcard/Videos"


def adb(*a, timeout=40):
    return sp.run(["adb", *a], capture_output=True, text=True, timeout=timeout).stdout


def tap(x, y):
    adb("shell", "input", "tap", str(x), str(y))


def ui():
    adb("shell", "uiautomator", "dump", "/sdcard/ui.xml", timeout=20)
    return adb("shell", "cat", "/sdcard/ui.xml", timeout=20)


def center(xml, exact=False, **attrs):
    """Center (x,y) of first node matching all attrs. exact=True -> exact text equality."""
    for chunk in xml.split("<node"):
        ok = True
        for k, v in attrs.items():
            m = re.search(rf'{k}="([^"]*)"', chunk)
            if not m:
                ok = False; break
            val = m.group(1)
            if (val != v) if exact else (v.lower() not in val.lower()):
                ok = False; break
        if ok:
            b = re.search(r'bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"', chunk)
            if b:
                x1, y1, x2, y2 = map(int, b.groups())
                return ((x1 + x2) // 2, (y1 + y2) // 2)
    return None


def list_videos():
    out = adb("shell", "ls", VIDEODIR).replace("\r", "")
    return set(f"{VIDEODIR}/{n}" for n in out.split("\n") if n.strip().endswith(".mp4"))


def download_one(vid, per_timeout=180):
    before = list_videos()
    adb("shell", "am", "start", "-n", "org.schabi.newpipe/.RouterActivity",
        "-a", "android.intent.action.VIEW", "-d", f"https://www.youtube.com/watch?v={vid}")

    def is_config(x):  # download config dialog
        return ("Threads" in x) or ("MPEG" in x) or ("Streams which" in x)

    # Poll for: config dialog (default handler set -> appears directly), OR the chooser
    # (Download radio -> tap it + "Just once"), OR an unavailable screen.
    opened = False
    for _ in range(16):
        xml = ui()
        if "not available" in xml.lower():
            return None, "unavailable"
        if is_config(xml):
            opened = True; break
        dl = center(xml, text="Download", **{"class": "RadioButton"})
        if dl:
            tap(*dl); time.sleep(1)
            x2 = ui()
            jo = center(x2, text="Just once") or center(x2, text="once")
            tap(*(jo or (770, 1640))); time.sleep(2)
        time.sleep(1.5)
    if not opened:
        return None, "no-dialog"
    # Confirm: tap OK, verify the config dialog actually closed; retry only while it's still
    # open (self-heals a missed tap without over-tapping onto the home screen behind it).
    for _ in range(4):
        tap(975, 934); time.sleep(2)
        x = ui()
        if "already exists" in x.lower():
            tap(*(center(x, text="Overwrite") or (835, 831))); time.sleep(2)
            break
        if not is_config(x):
            break

    # 4) wait for a NEW file to appear and finish (size stable)
    deadline = time.time() + per_timeout
    newf, last = None, -1
    while time.time() < deadline:
        cur = list_videos()
        fresh = list(cur - before)
        if fresh:
            newf = fresh[0]
            sz = adb("shell", "stat", "-c", "%s", newf).strip()
            if sz.isdigit() and int(sz) == last and int(sz) > 0:
                break
            last = int(sz) if sz.isdigit() else last
        time.sleep(2)
    if not newf:
        return None, "no-file"

    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, f"{vid}.mp4")
    adb("pull", newf, dst, timeout=180)
    adb("shell", "rm", newf)  # keep phone tidy
    return (dst, os.path.getsize(dst)) if os.path.exists(dst) else (None, "pull-failed")


def main():
    args = sys.argv[1:]
    requested = []
    for a in args:
        if a == "--file" or a.endswith(".txt"):
            continue
        requested += a.split()  # tolerate a single space-joined arg
    # --file <path>: read whitespace-separated IDs from a file
    if "--file" in args:
        path = args[args.index("--file") + 1]
        requested = open(path).read().split()
    elif any(a.endswith(".txt") for a in args):
        requested = open(next(a for a in args if a.endswith(".txt"))).read().split()
    ids = [v for v in requested if not os.path.exists(os.path.join(OUT, f"{v}.mp4"))]
    skipped = len(requested) - len(ids)
    print(f"downloading {len(ids)} new videos via NewPipe ({skipped} already in yt_downloads, skipped)...")
    ok = 0
    for i, vid in enumerate(ids, 1):
        try:
            res, info = download_one(vid)
        except Exception as e:
            res, info = None, f"{type(e).__name__}:{str(e)[:40]}"
        if res:
            ok += 1
            print(f"  [{i}/{len(ids)}] {vid}: OK ({info//1024//1024} MB)")
        else:
            print(f"  [{i}/{len(ids)}] {vid}: FAIL ({info})")
        time.sleep(2)
    print(f"done: {ok}/{len(ids)} downloaded to {OUT}")


if __name__ == "__main__":
    main()
