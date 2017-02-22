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