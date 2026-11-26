import argparse
from pathlib import Path

from vehicle_common import default_hef_path


def main():
    parser = argparse.ArgumentParser(description="Inspect the input and output streams of a Hailo HEF.")
    parser.add_argument("--hef-path", type=Path, default=default_hef_path())
    args = parser.parse_args()
    if not args.hef_path.is_file():
        parser.error(f"HEF not found: {args.hef_path}")
    from hailo_platform import HEF

    hef = HEF(str(args.hef_path))
    for name, streams in (("Input", hef.get_input_vstream_infos()), ("Output", hef.get_output_vstream_infos())):
        print(f"==== {name} vstream infos ====")
        for info in streams:
            print(f"{info.name}: shape={info.shape}, format={info.format}")


if __name__ == "__main__":
    main()
