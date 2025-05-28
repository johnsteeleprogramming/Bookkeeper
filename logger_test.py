import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    print(logger.log)
    print(logging.log)
    print(logging.getLogger(__name__))