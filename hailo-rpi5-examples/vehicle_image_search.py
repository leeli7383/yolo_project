"""Search Wikimedia Commons vehicle photos and compare Hailo detection thresholds."""
import argparse
import csv
import html
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request

import cv2
import numpy as np

from vehicle_common import (ROOT, VehicleDetector, add_model_options, annotate,
                            new_run_dir, positive, probability)

USER_AGENT = "VehicleDetectionDemo/1.0 (Hailo Raspberry Pi image evaluation)"


def fetch(url, limit=20 * 1024 * 1024):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("下载内容超过大小限制")
    return data


def search_images(query, count):
    params = dict(action="query", format="json", generator="search",
                  gsrsearch=f"{query} filetype:bitmap", gsrnamespace=6,
                  gsrlimit=min(50, count * 3), prop="imageinfo",
                  iiprop="url|extmetadata", iiurlwidth=1280)
    data = json.loads(fetch("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)))
    if "error" in data:
        raise RuntimeError(f"图片搜索失败：{data['error']}")
    pages = sorted(data.get("query", {}).get("pages", {}).values(), key=lambda page: page.get("index", 0))
    for page in pages:
        if not page.get("imageinfo"):
            continue
        info = page["imageinfo"][0]
        metadata = info.get("extmetadata", {})
        clean = lambda key: html.unescape(re.sub("<[^>]+>", "", metadata.get(key, {}).get("value", "")))
        yield dict(title=page["title"], url=info.get("thumburl", info["url"]),
                   source=info["descriptionurl"], author=clean("Artist"),
                   license=clean("LicenseShortName"), license_url=clean("LicenseUrl"))


def collect_images(args, run_dir):
    originals = run_dir / "originals"
    originals.mkdir()
    sources = []
    errors = []
    if args.images:
        candidates = (dict(title=Path(path).name, local=str(Path(path).resolve()),
                           source="", author="", license="", license_url="") for path in args.images)
    else:
        candidates = search_images(args.query, args.count)
    for candidate in candidates:
        try:
            raw = Path(candidate["local"]).read_bytes() if "local" in candidate else fetch(candidate["url"])
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("图片格式无法解码")
            if max(image.shape[:2]) > 1600:
                scale = 1600 / max(image.shape[:2])
                image = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
            name = f"image_{len(sources) + 1:03}.jpg"
            if not cv2.imwrite(str(originals / name), image):
                raise RuntimeError("图片写入失败")
            candidate["file"] = f"originals/{name}"
            sources.append(candidate)
            print(f"图片 {len(sources)}：{candidate['title']}", flush=True)
            if not args.images and len(sources) >= args.count:
                break
        except Exception as error:
            errors.append(dict(title=candidate["title"], error=str(error)))
            print(f"跳过 {candidate['title']}：{error}", flush=True)
    (run_dir / "sources.json").write_text(json.dumps({"images": sources, "errors": errors},
                                                   ensure_ascii=False, indent=2), encoding="utf-8")
    if not sources:
        raise RuntimeError("没有可用图片；请检查网络或使用 --images 本地图片路径")
    return sources


def write_report(run_dir, sources, results, args):
    escape = lambda value: html.escape(str(value), quote=True)
    sections = []
    for source in sources:
        cards = []
        for result in results:
            if result["original"] != source["file"]:
                continue
            labels = ", ".join(f"{d['label']} {d['confidence']:.3f}" for d in result["detections"]) or "未检测到车辆"
            cards.append(f'<figure><a href="{escape(result["annotated"])}"><img src="{escape(result["annotated"])}"></a>'
                         f'<figcaption>阈值 {result["threshold"]:.2f} · {len(result["detections"])} 个车辆<br>{escape(labels)}</figcaption></figure>')
        source_link = f'<a href="{escape(source["source"])}">图片来源</a>' if source["source"] else "本地图片"
        sections.append(f'<section><h2>{escape(source["title"])}</h2><p>{source_link} · {escape(source["author"])} · '
                        f'{escape(source["license"])} · <a href="{escape(source["file"])}">原图</a></p>'
                        f'<div class="grid">{"".join(cards)}</div></section>')
    document = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width, initial-scale=1"><title>车辆识别效果对比</title>'
                '<style>body{font:16px system-ui;margin:24px;background:#f4f6f8;color:#172435}'
                '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}'
                'figure{margin:0;background:white;padding:12px;border-radius:12px}img{width:100%}'
                'figcaption{line-height:1.8;overflow-wrap:anywhere}section{margin-bottom:32px}</style>'
                '<h1>车辆识别效果对比</h1><p>框旁数字是模型置信度；阈值是保留检测框的最低置信度。'
                '未检测到车辆不代表图片中一定没有车辆。点击图片查看大图。</p>'
                f'<p>模型：{escape(args.hef_path.name)} · NMS IoU：{args.iou} · 类别：{escape(", ".join(args.classes))}</p>'
                + "".join(sections) + '</html>')
    (run_dir / "report.html").write_text(document, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_options(parser)
    parser.add_argument("--query", default="car traffic", help="网络图片关键词，英文通常结果更多")
    parser.add_argument("--count", type=positive, default=3)
    parser.add_argument("--images", nargs="+", help="使用本地图片，跳过网络搜索")
    parser.add_argument("--thresholds", nargs="+", type=probability, default=[0.25, 0.50, 0.75])
    parser.add_argument("--output-dir", default=ROOT / "records/vehicle_images")
    args = parser.parse_args()
    if args.count > 50:
        parser.error("--count 最大为 50")
    thresholds = sorted(set(args.thresholds))
    run_dir = new_run_dir(args.output_dir)
    print(f"输出目录：{run_dir}", flush=True)
    sources = collect_images(args, run_dir)
    results = []
    (run_dir / "annotated").mkdir()
    with VehicleDetector(args.hef_path, min(thresholds), args.iou, args.classes) as detector:
        with (run_dir / "detections.csv").open("w", newline="", encoding="utf-8") as file:
            log = csv.writer(file)
            log.writerow(["image", "threshold", "label", "confidence", "x1", "y1", "x2", "y2"])
            for source in sources:
                image = cv2.imread(str(run_dir / source["file"]))
                detections = detector.detect(image)
                for threshold in thresholds:
                    selected = [det for det in detections if det["confidence"] >= threshold]
                    caption = f"threshold={threshold:.3f} | NMS IoU={args.iou:.2f} | vehicles={len(selected)}"
                    annotated = annotate(image, selected, caption)
                    name = f"annotated/{Path(source['file']).stem}_threshold_{threshold:g}.jpg"
                    if not cv2.imwrite(str(run_dir / name), annotated):
                        raise RuntimeError(f"保存结果失败：{name}")
                    results.append(dict(original=source["file"], annotated=name,
                                        threshold=threshold, detections=selected))
                    for det in selected:
                        log.writerow([source["file"], threshold, det["label"], det["confidence"], *det["box"]])
                    print(f"{source['title']} | {caption}", flush=True)
    (run_dir / "results.json").write_text(json.dumps({"hef": str(args.hef_path), "iou": args.iou,
        "classes": args.classes, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(run_dir, sources, results, args)
    print(f"完成。用浏览器打开对比报告：{run_dir / 'report.html'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"车辆图片识别失败：{error}") from error
