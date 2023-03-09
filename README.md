# Petris

Tetris clone with AI agents to play the game.

# Requirements

-   Python3.8 >=

## Set Python Virtual Environment

Create `venv` Directory:

```bash
python3 -m venv ./venv
```

Activate virtual environment:

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip3 install -r requirements.txt
```

# Startup

```bash
python3 src/petris.py
```

## Optional Arguments

`-s/--speed`: Sets the speed of the clock. Higher the faster the piece falls down.

## Example Arguments

```bash
python3 src/petris.py --speed 100
```

# Acknowledgments

-   https://docs.python.org/3/library/venv.html
