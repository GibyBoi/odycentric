# Odycentric

A Windows desktop app that removes backgrounds from photos, on your own PC, with no limits and nothing uploaded.

Drop in photos or whole folders, pick your settings, and it saves transparent PNGs. While it works it takes over
the processor (you choose how much), and every photo gets a score so the app can suggest the best settings for
the situation you are in.

![Odycentric](assets/odycentric.png)

## Why not an online remover

Tools like ChatGPT "remove the background" by regenerating the whole picture, which is why text on shirts, signs
and logos comes back garbled. Odycentric never redraws anything. The AI only decides how see-through each pixel
is. Every pixel of your subject that is fully solid is copied from the original byte for byte (there is a test
for this), and the removed background is erased from the file, not just hidden.

## Install

Needs Windows 10 or 11 and Python 3.10+ (python.org or the Microsoft Store).

```powershell
git clone https://github.com/GibyBoi/odycentric.git
cd odycentric
powershell -ExecutionPolicy Bypass -File setup.ps1
```

That creates a private Python environment in `.venv` and an **Odycentric** shortcut on your desktop. Open it, or
drop photos straight onto the shortcut. Each AI model is downloaded the first time you use it (0.2 to 1.1 GB,
from the [rembg](https://github.com/danielgatis/rembg) releases) and checked against a SHA-256 fingerprint.

## Settings

| Group | Setting | What it does |
|---|---|---|
| Cutout | Subject | **People** (portrait-trained), **Anything**, **Fast** (small, about twice as quick), and two 2048 px **HD** models |
| | AI resolution | The model's own size (1024 or 2048) is one pass over the whole photo. Higher (up to 4096) adds detail passes along the subject's outline. Several times slower, same memory |
| | Edges | Soft keeps natural see-through hair and fur; Sharp makes every pixel solid or clear |
| | Clean edge colours | Takes the old background's tint out of see-through edge pixels so hair has no halo |
| Output | Background | Transparent, white, black or any colour |
| | Size | Original, or shrink to 4096 / 2048 / 1024 px first |
| Performance | Threads | Auto (one per core), all threads, or a fixed number |
| | Priority | High takes over the PC until the photos are done; Normal shares; Low uses spare time only |
| | Memory limit | Auto is all of your RAM but 3 GB |

## Scores and suggestions

Every finished photo gets a score from 0 to 100 that mixes quality and speed:

- **Quality** is your star rating if you gave one (5 stars = 100). Otherwise it is **edge clarity**: how thin the
  see-through band around the subject's outline is, measured on a 1024 px grid (a band of 2 px or less is 100).
- **Speed** is 100 at 10 seconds per photo or less, and halves each time the time doubles.
- **Optimize for** sets the mix: Quality 80/20, Balanced 60/40, Speed 30/70.

Suggestions only use photos done in the same power situation (plugged in or on battery) once there are three of
them, because the same settings run at very different speeds in each. Threads and priority never change the
cutout, so quality is pooled across them and they are judged on speed alone. A setting tried on a single photo
needs a clear lead to beat a steady record. History lives in `data/history.sqlite`.

## How it keeps your PC alive

The AI runs in a separate worker process inside a Windows Job Object:

- **Memory cap.** The worker can never use more than the memory limit. A photo that needs more fails on its own
  instead of the PC freezing. The app also checks free memory before loading a model and stops if Windows gets
  down to its last 0.5 GB.
- **Kill on close.** If the app closes or crashes, Windows ends the worker with it.
- **Stop is instant.** Stop kills the worker on the spot; waiting photos stay queued.
- **CPU only, on purpose.** Running these models on an integrated GPU (Intel Arc through DirectML) hard-froze the
  development laptop, because that GPU also draws the screen and shares system RAM. The GPU is never used.

Measured on an Intel Core Ultra 7 155H, plugged in, High priority: the People model takes about 6 s to load and
14 s per photo at 1024 px, peaking at about 7.4 GB of memory; a 2048 px detail run took about 95 s on an 8.5 MP
portrait at the same memory. The HD models need an estimated 22 GB and have not been run end to end on a
machine with that much free memory yet.

## Development

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -t .
.venv\Scripts\python.exe -m odycentric
```

`engine.py` is the cutout itself (no Qt), `worker.py` runs it in the worker process, `guard.py` is the Job Object,
`history.py` the scores, and `app.py` the window.

## Credits

Models: [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) by Zheng Peng et al. (MIT licence), ONNX exports
from [rembg](https://github.com/danielgatis/rembg). Edge colour cleanup is the blur-fusion foreground estimator
(Forte and Pitié, 2021), as used in BiRefNet. Test photos in development were public-domain NASA images.
