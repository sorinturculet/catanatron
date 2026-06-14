# Catanatron

> **Fork of**: [bcollazo/catanatron](https://github.com/bcollazo/catanatron) - Settlers of Catan simulator

## Diploma Thesis — Imitation-Initialized RL for Settlers of Catan

This fork contains the experimental code, training pipelines, and web application accompanying the diploma thesis:

> **Imitation-Initialized Reinforcement Learning for Strategic Decision-Making in Settlers of Catan**
> Turculeț Sorin-Nicolae — Babeș-Bolyai University, Faculty of Mathematics and Computer Science, 2026
> Scientific supervisor: Prof. dr. Gabriela Czibula

The full thesis PDF is included in the root of this repository: [Thesis_Turculet_Sorin.pdf](Thesis_Turculet_Sorin.pdf). Detailed per-version evaluation metrics are stored in [ppo_eval_runs.csv](ppo_eval_runs.csv).

### Methodology

Three policy-optimization pipelines were evaluated, all built on Maskable PPO ([Stable-Baselines3](https://stable-baselines3.readthedocs.io)) with a CNN feature extractor over the spatial board representation. All evaluations are 1v1 against the heuristic ValueFunctionPlayer (VFP) over 500 games.

1. **Pure-PPO** — trained from scratch with environment rewards only.
2. **EG-PPO (Expert-Guided)** — PPO from scratch with a continuous shaping bonus derived from the VFP heuristic score.
3. **BC-PPO (Behavioral Cloning + PPO)** — supervised pre-training of the policy on 100K VFP self-play games (~15M decision points), followed by unconstrained PPO fine-tuning with no expert wrapper.

### Main Results

Win rate against the ValueFunctionPlayer baseline (500 games per row):

| Pipeline | Version | Training Steps | Win Rate | Avg VP |
|----------|---------|----------------|----------|--------|
| Pure-PPO | v2 (CNN + MLP, dense VP rewards) | 10M | 8.2% | 5.15 |
| Pure-PPO | v6 (GNN GATv2) | 10M | 9.6% | 5.31 |
| Pure-PPO | v9 (cyclic self-play from scratch) | 10M | 3.0% | 3.95 |
| EG-PPO | v15 (expert guidance + production-aware) | 10M | 15.4% | 6.16 |
| EG-PPO | v19 (extended training) | 30M | 21.2% | 7.03 |
| EG-PPO | **v26** (5-stage LR, road/settle bonus) | 50M | **31.2%** | 7.23 |
| BC-PPO | v34 (50K-game BC + dense VP rewards) | 50M | 60.0% | 8.63 |
| BC-PPO | v35 (100K-game BC, cosine LR) | 80M | 66.4% | 8.95 |
| BC-PPO | v36 (SGDR warm restart from v35) | 120M | 68.8% | 8.99 |
| BC-PPO | **v37** (mixed VFP opponents, prod-aware) | 170M | **69.6%** | 9.08 |

The final BC-PPO agent (v37) wins **348 of 500** head-to-head games against the heuristic baseline.

### Key Findings

- **Pure-PPO hits a ~10% ceiling.** Neither architectural changes (GNN, attention) nor reward shaping alone broke through. Credit assignment over ~400 sequential decisions with sparse terminal rewards is the bottleneck, not network capacity.
- **Expert guidance breaks the ceiling but creates a new one.** EG-PPO reached 31.2% at 50M steps. The shaping signal biases the agent toward the 1-ply heuristic's strategy and penalizes superior non-heuristic moves, capping further improvement.
- **Behavioral cloning decouples knowledge from the RL objective.** Initializing the policy with supervised pre-training and removing the shaping wrapper during fine-tuning more than doubled the EG-PPO ceiling, reaching 69.6%.
- **Paradox of teacher quality.** Pre-training on the strategically stronger AlphaBeta(depth=2) teacher *degrades* final performance compared to pre-training on the simpler VFP heuristic:

  | Model | BC Teacher | PPO Opponent | vs VFP | vs AlphaBeta |
  |-------|-----------|--------------|--------|--------------|
  | v35 | VFP (100K games) | VFP | **66.4%** | **65.4%** |
  | v38 | AlphaBeta (100K games) | VFP | 61.6% | 60.0% |
  | v39 | AlphaBeta (100K games) | AlphaBeta | 57.8% | 56.4% |

  The 1-ply policy network cannot reconstruct AlphaBeta's multi-ply search tree from static observations, so the network sees the teacher's actions as noisy with respect to the immediate state. Teacher–student architectural compatibility matters more than raw teacher strength.

### Reproducing the Experiments

Training scripts live in [catanatron_experimental/catanatron_experimental/machine_learning/](catanatron_experimental/catanatron_experimental/machine_learning/), one per version (`train_ppo_agent_*.py`, `train_ablation_*.py`). The behavioral cloning data collectors are `collect_and_train_bc.py` (VFP teacher) and `collect_and_train_bc_alphabeta.py` (AlphaBeta teacher). Trained checkpoints are not committed to the repo because of GitHub file-size limits — they are produced by the training scripts.

The PPO agent itself is registered as a player under [catanatron_experimental/.../players/ppo.py](catanatron_experimental/catanatron_experimental/machine_learning/players/ppo.py) and can be played via the CLI:

```
catanatron-play --players=PPO,VP --num=500
```

### Web Application

The thesis also includes a web application (Chapter 4) for playing against the trained agents in a browser. The original Catanatron repository provides a polling-based React/Flask stack; the thesis refactor adds a WebSocket pipeline, a dynamic Stable-Baselines3 model registry, JWT-based user accounts with PostgreSQL persistence, and updated SVG board assets. See [docker-compose.yml](docker-compose.yml) and the original instructions below for running the full stack locally.

---

[![Coverage Status](https://coveralls.io/repos/github/bcollazo/catanatron/badge.svg?branch=master)](https://coveralls.io/github/bcollazo/catanatron?branch=master)
[![Documentation Status](https://readthedocs.org/projects/catanatron/badge/?version=latest)](https://catanatron.readthedocs.io/en/latest/?badge=latest)
[![Join the chat at https://gitter.im/bcollazo-catanatron/community](https://badges.gitter.im/bcollazo-catanatron/community.svg)](https://gitter.im/bcollazo-catanatron/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bcollazo/catanatron/blob/master/catanatron_experimental/catanatron_experimental/Overview.ipynb)

Settlers of Catan Bot and Bot Simulator. Test out bot strategies at scale (thousands of games per minutes). The goal of this project is to find the strongest Settlers of Catan bot possible.

See the motivation of the project here: [5 Ways NOT to Build a Catan AI](https://medium.com/@bcollazo2010/5-ways-not-to-build-a-catan-ai-e01bc491af17).

<p align="left">
 <img src="https://raw.githubusercontent.com/bcollazo/catanatron/master/docs/source/_static/cli.gif">
</p>

## Installation

Clone this repository and install dependencies. This will include the Catanatron bot implementation and the `catanatron-play` simulator.

```
git clone git@github.com:bcollazo/catanatron.git
cd catanatron/
```

Create a virtual environment with Python3.8 or higher. Then:

```
pip install -r all-requirements.txt
```

## Usage

Run simulations and generate datasets via the CLI:

```
catanatron-play --players=R,R,R,W --num=100
```

See more information with `catanatron-play --help`.

## Try Your Own Bots

Implement your own bots by creating a file (e.g. `myplayers.py`) with some `Player` implementations:

```python
from catanatron import Player
from catanatron_experimental.cli.cli_players import register_player

@register_player("FOO")
class FooPlayer(Player):
  def decide(self, game, playable_actions):
    """Should return one of the playable_actions.

    Args:
        game (Game): complete game state. read-only.
        playable_actions (Iterable[Action]): options to choose from
    Return:
        action (Action): Chosen element of playable_actions
    """
    # ===== YOUR CODE HERE =====
    # As an example we simply return the first action:
    return playable_actions[0]
    # ===== END YOUR CODE =====
```

Run it by passing the source code to `catanatron-play`:

```
catanatron-play --code=myplayers.py --players=R,R,R,FOO --num=10
```

## How to Make Catanatron Stronger?

The best bot right now is Alpha Beta Search with a hand-crafted value function. One of the most promising ways of improving Catanatron
is to have your custom player inhert from ([`AlphaBetaPlayer`](catanatron_experimental/catanatron_experimental/machine_learning/players/minimax.py)) and set a better set of weights for the value function. You can
also edit the value function and come up with your own innovative features!

For more sophisticated approaches, see example player implementations in [catanatron_core/catanatron/players](catanatron_core/catanatron/players)

If you find a bot that consistently beats the best bot right now, please submit a Pull Request! :)

## Advanced Usage

### Inspecting Games (Browser UI)

We provide a [docker-compose.yml](docker-compose.yml) with everything needed to watch games (useful for debugging). It contains all the web-server infrastructure needed to render a game in a browser.

<p align="left">
 <img src="https://raw.githubusercontent.com/bcollazo/catanatron/master/docs/source/_static/CatanatronUI.png">
</p>

To use, ensure you have [Docker Compose](https://docs.docker.com/compose/install/) installed, and run (from this repo's root):

```
docker-compose up
```

You can now use the `--db` flag to make the catanatron-play simulator save
the game in the database for inspection via the web server.

```
catanatron-play --players=W,W,W,W --db --num=1
```

NOTE: A great contribution would be to make the Web UI allow to step forwards and backwards in a game to inspect it (ala chess.com).

### Accumulators

The `Accumulator` class allows you to hook into important events during simulations.

For example, write a file like `mycode.py` and have:

```python
from catanatron import ActionType
from catanatron_experimental import SimulationAccumulator, register_accumulator

@register_accumulator
class PortTradeCounter(SimulationAccumulator):
  def before_all(self):
    self.num_trades = 0

  def step(self, game_before_action, action):
    if action.action_type == ActionType.MARITIME_TRADE:
      self.num_trades += 1

  def after_all(self):
    print(f'There were {self.num_trades} port trades!')
```

Then `catanatron-play --code=mycode.py` will count the number of trades in all simulations.

### As a Package / Library

You can also use `catanatron` package directly which provides a core
implementation of the Settlers of Catan game logic.

```python
from catanatron import Game, RandomPlayer, Color

# Play a simple 4v4 game
players = [
    RandomPlayer(Color.RED),
    RandomPlayer(Color.BLUE),
    RandomPlayer(Color.WHITE),
    RandomPlayer(Color.ORANGE),
]
game = Game(players)
print(game.play())  # returns winning color
```

You can use the `open_link` helper function to open up the game (useful for debugging):

```python
from catanatron_server.utils import open_link
open_link(game)  # opens game in browser
```

## Architecture

The code is divided in the following 5 components (folders):

- **catanatron**: A pure python implementation of the game logic. Uses `networkx` for fast graph operations. Is pip-installable (see `setup.py`) and can be used as a Python package. See the documentation for the package here: https://catanatron.readthedocs.io/.

- **catanatron_server**: Contains a Flask web server in order to serve
  game states from a database to a Web UI. The idea of using a database, is to ease watching games played in a different process. It defaults to using an ephemeral in-memory sqlite database. Also pip-installable (not publised in PyPi however).

- **catanatron_gym**: OpenAI Gym interface to Catan. Includes a 1v1 environment against a Random Bot and a vector-friendly representations of states and actions. This can be pip-installed independently with `pip install catanatron_gym`, for more information see [catanatron_gym/README.md](catanatron_gym/README.md).

- **catantron_experimental**: A collection of unorganized scripts with contain many failed attempts at finding the best possible bot. Its ok to break these scripts. Its pip-installable. Exposes a `catanatron-play` command-line script that can be used to play games in bulk, create machine learning datasets of games, and more!

- **ui**: A React web UI to render games. This is helpful for debugging the core implementation. We decided to use the browser as a randering engine (as opposed to the terminal or a desktop GUI) because of HTML/CSS's ubiquitousness and the ability to use modern animation libraries in the future (https://www.framer.com/motion/ or https://www.react-spring.io/).

## AI Bots Leaderboard

Catanatron will always be the best bot in this leaderboard.

The best bot right now is `AlphaBetaPlayer` with n = 2. Here a list of bots strength. Experiments
done by running 1000 (when possible) 1v1 games against previous in list.

| Player               | % of wins in 1v1 games      | num games used for result |
| -------------------- | --------------------------- | ------------------------- |
| AlphaBeta(n=2)       | 80% vs ValueFunction        | 25                        |
| ValueFunction        | 90% vs GreedyPlayouts(n=25) | 25                        |
| GreedyPlayouts(n=25) | 100% vs MCTS(n=100)         | 25                        |
| MCTS(n=100)          | 60% vs WeightedRandom       | 15                        |
| WeightedRandom       | 53% vs WeightedRandom       | 1000                      |
| VictoryPoint         | 60% vs Random               | 1000                      |
| Random               | -                           | -                         |

## Developing for Catanatron

To develop for Catanatron core logic you can use the following test suite:

```
coverage run --source=catanatron -m pytest tests/ && coverage report
```

Or you can run the suite in watch-mode with:

```
ptw --ignore=tests/integration_tests/ --nobeep
```

## Machine Learning

Generate JSON files with complete information about games and decisions by running:

```
catanatron-play --num=100 --output=my-data-path/ --json
```

Similarly (with Tensorflow installed) you can generate several GZIP CSVs of a basic set of features:

```
catanatron-play --num=100 --output=my-data-path/ --csv
```

You can then use this data to build a machine learning model, and then
implement a `Player` subclass that implements the corresponding "predict"
step of your model. There are some attempts of these type of
players in [reinforcement.py](catanatron_experimental/catanatron_experimental/machine_learning/players/reinforcement.py).

# Appendix

## Running Components Individually

As an alternative to running the project with Docker, you can run the following 3 components: a React UI, a Flask Web Server, and a PostgreSQL database in three separate Terminal tabs.

### React UI

```
cd ui/
npm install
npm start
```

This can also be run via Docker independetly like (after building):

```
docker build -t bcollazo/catanatron-react-ui:latest ui/
docker run -it -p 3000:3000 bcollazo/catanatron-react-ui
```

### Flask Web Server

Ensure you are inside a virtual environment with all dependencies installed and
use `flask run`.

```
python3.8 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt

cd catanatron_server/catanatron_server
flask run
```

This can also be run via Docker independetly like (after building):

```
docker build -t bcollazo/catanatron-server:latest . -f Dockerfile.web
docker run -it -p 5000:5000 bcollazo/catanatron-server
```

### PostgreSQL Database

Make sure you have `docker-compose` installed (https://docs.docker.com/compose/install/).

```
docker-compose up
```

Or run any other database deployment (locally or in the cloud).

## Other Useful Commands

### TensorBoard

For watching training progress, use `keras.callbacks.TensorBoard` and open TensorBoard:

```
tensorboard --logdir logs
```

### Docker GPU TensorFlow

```
docker run -it tensorflow/tensorflow:latest-gpu-jupyter bash
docker run -it --rm -v $(realpath ./notebooks):/tf/notebooks -p 8888:8888 tensorflow/tensorflow:latest-gpu-jupyter
```

### Testing Performance

```
python -m cProfile -o profile.pstats catanatron_experimental/catanatron_experimental/play.py --num=5
snakeviz profile.pstats
```

```
pytest --benchmark-compare=0001 --benchmark-compare-fail=mean:10% --benchmark-columns=min,max,mean,stddev
```

### Head Large Datasets with Pandas

```
In [1]: import pandas as pd
In [2]: x = pd.read_csv("data/mcts-playouts-labeling-2/labels.csv.gzip", compression="gzip", iterator=True)
In [3]: x.get_chunk(10)
```

### Publishing to PyPi

catanatron Package

```
make build PACKAGE=catanatron_core
make upload PACKAGE=catanatron_core
make upload-production PACKAGE=catanatron_core
```

catanatron_gym Package

```
make build PACKAGE=catanatron_gym
make upload PACKAGE=catanatron_gym
make upload-production PACKAGE=catanatron_gym
```

### Building Docs

```
sphinx-quickstart docs
sphinx-apidoc -o docs/source catanatron_core
sphinx-build -b html docs/source/ docs/build/html
```

# Contributing

I am new to Open Source Development, so open to suggestions on this section. The best contributions would be to make the core bot stronger.

Other than that here is also a list of ideas:

- Improve `catanatron` package running time performance.

  - Continue refactoring the State to be more and more like a primitive `dict` or `array`.
    (Copies are much faster if State is just a native python object).
  - Move RESOURCE to be ints. Python `enums` turned out to be slow for hashing and using.
  - Move .actions to a Game concept. (to avoid copying when copying State)
  - Remove .current_prompt. It seems its redundant with (is_moving_knight, etc...) and not needed.

- Improve AlphaBetaPlayer:

  - Explore and improve prunning
  - Use Bayesian Methods or SPSA to tune weights and find better ones.

- Experiment ideas:

  - DQN Render Method. Use models/mbs=64\_\_1619973412.model. Try to understand it.
  - DQN Two Layer Algo. With Simple Action Space.
  - Simple Alpha Go
  - Try Tensorforce with simple action space.
  - Try simple flat CSV approach but with AlphaBeta-generated games.
  - Visualize tree with graphviz. With colors per maximizing/minimizing.
  - Create simple entry-point notebook for this project. Runnable via Paperspace. (might be hard because catanatron requires Python 3.8 and I haven't seen a GPU-enabled tensorflow+jupyter+pyhon3.8 Docker Image out there).

- Bugs:

  - Shouldn't be able to use dev card just bought.

- Features:

  - Continue implementing actions from the UI (not all implemented).
  - Chess.com-like UI for watching game replays (with Play/Pause and Back/Forward).
  - A terminal UI? (for ease of debugging)
