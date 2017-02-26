# Chessbot
Chessbot is a real-time chess assistant using:
- computer vision (OpenCV)
- chess library (python-chess)
- chess engine (Stockfish)
  
Works best with:
- a high-resolution webcam
- webcam placed straight above the chess board
- a chess set with 4 distinct colours:
    - light squares
    - dark squares
    - light (white) pieces
    - dark (black) pieces

GUI  
![GUI](/images/screenshot-1.png)

## Setup
```
git clone https://github.com/cornsmith/chessbot.git  
cd chessbot  
mkdir calibration  
git clone https://github.com/official-stockfish/Stockfish.git  
cd Stockfish/src  
make profile-build ARCH=x8 
```

Alternatively use the docker image
```
docker build -t chessbot .
. Dockerrun.sh
```

## Equipment
- Webcam
- Stand
- Chessset

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

### Using test images / demo
```
python chessbot/chessbot.py --calibrate=1 --camera=-1 
python chessbot/chessbot.py --calibrate=2 --camera=-1
python chessbot/chessbot.py --camera=-1
```


## Screenshots
Equipment  
![Equipment](/images/setup.jpg)

Calibration  
![Calibration](/images/screenshot-calibration-1.png)
