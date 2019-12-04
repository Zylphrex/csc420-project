# CSC420 Project


## How to run

```
git clone --depth 1 git@github.com:Zylphrex/csc420-project.git
cd csc420-project && git submodule init && git pull --recurse-submodules
# download pre-trained model on dropbox from https://github.com/meijieru/crnn.pytorch and save it as crnn.pth in the root directory of this repository
python main.py <image>
```

### PLEASE READ

Try the test images in the `images` folder. To use your own images, make sure they are of size 2268 X 4032 or 4032 X 2268 and the document is approximately vertical in the image.
