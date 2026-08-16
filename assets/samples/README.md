# Bundled demo samples

Anything dropped in this folder appears as a one-click button on the dashboard,
so a visitor can try the system without finding their own photo. `app.py`
discovers files by scanning this directory — **no code change is needed** to add
or remove a sample.

## Naming

`NN_label_words.ext` — the `NN_` prefix orders the buttons and is stripped from
the label, and underscores become spaces. So:

| File | Button |
|---|---|
| `01_standing.jpg` | Standing |
| `02_bending_over.jpg` | Bending over |
| `03_fallen.jpg` | Fallen |
| `04_fall_clip.mp4` | Fall clip |

Images (`.jpg .jpeg .png .bmp .webp`) go to the Image Analysis tab; videos
(`.mp4 .mov .avi .mkv .webm`) go to Video Monitoring.

## What makes a good sample

The samples are processed by MediaPipe exactly like an upload, so they have to
be real photographs of a person — a synthetic or empty image will correctly
return "no person detected".

* **whole body in frame, head to feet** — the single most common reason pose
  estimation fails is cropped feet
* one person only; a second subject is ignored
* even lighting, uncluttered background
* shoot the bending sample **side-on**, or the trunk angle is invisible
* shoot the fallen sample from roughly chest height angled down, mimicking a
  wall-mounted monitor
* keep clips to 10–20 s and under 60 MB; frames are sampled at 6 fps

## Privacy

These files are committed to a public repository and served to anyone who opens
the app. Only use footage of people who have agreed to that.
