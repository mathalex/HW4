import os
import shutil
from pathlib import Path

import gdown

URL_LINKS = {
    "train": "https://drive.google.com/u/0/uc?id=16KyLIbXUNjk8y5iXSxNt0Jkr7-2xbYro&export=download"
}

def download():
    data_dir = Path(__file__).absolute().resolve().parent.parent
    data_dir = data_dir / "saved" / "models" / "pretrained"
    data_dir.mkdir(exist_ok=True, parents=True)

    arc_path = data_dir / 'train.zip'
    if not arc_path.exists():
        gdown.download(URL_LINKS['train'], str(arc_path))
    shutil.unpack_archive(str(arc_path), str(data_dir))
    os.remove(str(arc_path))

if __name__ == '__main__':
    download()
