"""Logging utils."""

import time
from functools import wraps


def log_function_info(func):
    """Log decorator which shows running function and execution time.

    Args:
        func (_type_): func which we log

    Raises:
        e: error

    Returns:
        _type_: func
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        log_start_info = f"### Start {func.__qualname__} ###"
        hashtag_line = "#" * len(log_start_info)

        self.logger.info(hashtag_line)
        self.logger.info(log_start_info)
        self.logger.info(hashtag_line)

        try:
            result = func(self, *args, **kwargs)
            success = True
            return result
        except Exception as e:
            success = False
            raise e  # Re-raise the caught exception after logging
        finally:
            end_time = time.time()
            duration = end_time - start_time

            if success:
                log_end_info = (
                    f"### {func.__qualname__} completed. Execution time: {duration:.4f} seconds ###"
                )
            else:
                log_end_info = (
                    f"### {func.__qualname__} failed. Execution time: {duration:.4f} seconds ###"
                )

            hashtag_line = "#" * len(log_end_info)

            self.logger.info(hashtag_line)
            self.logger.info(log_end_info)
            self.logger.info(hashtag_line)

    return wrapper
