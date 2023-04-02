"""
Another main script for training the avaialable agents.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# NOTE: Going to set the paths in here before importing packages. 
# NOTE: This is important for importing packages.
sys.path.append(str(Path().parent))

from src import paths
from src.log.log import initialize_logger
from src.agents.reinforce_agent import train_reinforce

logger = logging.getLogger(__name__)

TRAIN_LOG_FILE = "train.log"
TRAIN_LOG_PATH = paths.LOG_PATH / TRAIN_LOG_FILE

# TODO: Need to use pickle to serialize the trained model for the other main script to reuse.


def main(agent: str, debug: bool = False) -> int:
    """
    Main function that handles all exceptions and begins the training process.
    Args:
        agent: The agent to train.
        debug: Flag to display debug logs
    Returns:
        int: Exit code
    """
    exit_code = 0

    try:
        initialize_logger(log_path=TRAIN_LOG_PATH, debug=debug)

        logger.info("Beginning Training Agent: %s", agent.upper())

        if agent.lower() == "reinforce":
            train_reinforce()
    except Exception as ex:
        exit_code = 1
        logger.exception(ex)
    finally:
        logger.info("End of Training: %s", exit_code)

    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--agent", action="store", required=True, type=str,
                        help="Agent to train.")
    parser.add_argument("-d", "--debug", action="store_true", required=False, default=False,
                        help="Displays the debug logs.")

    args, _ = parser.parse_known_args()

    sys.exit(main(agent=args.agent, debug=args.debug))