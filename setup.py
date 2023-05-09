from setuptools import setup, find_packages
from tracknetv2 import __version__ as version

setup_params = {
    "name": "tracknetv2",
    "version": version,
    "description": "An unofficial implementation of TrackNetV2 from 'TrackNetV2: Efficient Shuttlecock \
    Tracking Network' (2020) by Nien-En Sun, Yu-Ching Lin et al.",
    "url": "https://github.com/ron1x1-abba/PyTorchTrackNetv2",
    "author": "Golikov Artemii",
    "license": "MIT License",
    "packages": find_packages(),
    "include_package_data": True,
    "zip_unsafe": False,
    "install_requires": []
}

if __name__ == "__main__":
    setup(**setup_params)
