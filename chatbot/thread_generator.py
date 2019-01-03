#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import queue
import threading


class ThreadGenerator(object):

    def __init__(self, iterator, sentinel=object(), queue_maxsize=0, daemon=False):
        self._iterator = iterator
        self._sentinel = sentinel
        self._queue = queue.Queue(maxsize=queue_maxsize)
        self._thread = threading.Thread(name=repr(iterator), target=self._run)
        self._thread.daemon = daemon
        self._started = False

    def __repr__(self):
        return 'ThreadedGenerator({})'.format(self._iterator)

    def _run(self):
        try:
            for value in self._iterator:
                if not self._started:
                    return
                self._queue.put(value)
        finally:
            self._queue.put(self._sentinel)

    def close(self):
        self._started = False
        try:
            while True:
                self._queue.get(timeout=30)
        except KeyboardInterrupt as e:
            raise e
        except:
            pass

    def __iter__(self):
        self._started = True
        self._thread.start()
        for value in iter(self._queue.get, self._sentinel):
            yield value
        self._thread.join()
        self._started = False

    def __next__(self):
        if not self._started:
            self._started = True
            self._thread.start()
        value = self._queue.get(timeout=30)
        if value == self._sentinel:
            raise StopIteration
        return value


def test():
    def gene():
        i = 0
        while True:
            yield i
            i += 1
    t = gene()
    test = ThreadGenerator(iterator=t)
    for _ in range(20):
        print(next(test))
    test.close()


if __name__ == '__main__':
    test()