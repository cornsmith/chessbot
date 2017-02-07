## Setup
TODO
git clone ...
mkdir calibration
mkdir Stockfish
...

## Usage
Calibration - 2 steps
```
python chessbot/chessbot.py --calibrate 1 
python chessbot/chessbot.py --calibrate 2
```

Play
```
python chessbot/chessbot.py
```

### Using test images

```
python chessbot/chessbot.py --calibrate=1 --camera=-1 
python chessbot/chessbot.py --calibrate=2 --camera=-1
python chessbot/chessbot.py --camera=-1
```