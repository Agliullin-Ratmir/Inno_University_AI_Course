import unittest
from unittest.mock import Mock

class ThresholdReached(Exception):
    def __init__(self, threshold):
        super().__init__(f"Threshold has bee reached")


def count_calls(threshold=1):
    def decorator(func):
        func.call_count = 0

        def wrapper(*args, **kwargs):
            func.call_count = func.call_count + 1
            if func.call_count > threshold:
                raise ThresholdReached(threshold)
            return func(*args, **kwargs)
        return wrapper

    return decorator

class CallsTester(unittest.TestCase):

    def test(self):
        @count_calls(threshold=2)
        def test_calls():
            return True

        # Должно сработать 3 раза без исключения
        self.assertEqual(test_calls(), True)
        self.assertEqual(test_calls(), True)
        with self.assertRaises(ThresholdReached):
            test_calls()