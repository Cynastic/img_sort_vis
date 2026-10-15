import sys

from visualizer import run


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else "photo.jpg"
    run(img_path, "Gnome sort", do_shuffle=True, realtime=True)


if __name__ == "__main__":
    main()
