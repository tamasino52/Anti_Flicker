# Abstract
Program to eliminate flicker phenomena in high-speed video

## Defendency
```
pip install -r requirements.txt
```

## Run
```
python main.py -v "./video/C0183.MP4" -i "./interim" -o "./output"
```

## Test
<b>Original Video</b><br>
<img src="docs/input.gif"></img><p>
<b>Interim Video (Luminance Invariant Restoration)</b><br>
<img src="docs/interim.gif"></img><p>
<b>Output Video</b><br>
<img src="docs/output.gif"></img><p>
