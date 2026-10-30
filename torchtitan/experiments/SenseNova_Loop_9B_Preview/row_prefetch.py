# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Bounded speculative CPU row encoding, without advancing the data cursor."""

from concurrent.futures import ThreadPoolExecutor


class RowEncodingPrefetch:
    """Run every encode on one worker; consume results on the training thread.

    The encoder must be deterministic, CPU-only and independent of loader state.
    In particular, it must not issue distributed collectives or mutate the cursor.
    """

    def __init__(self, encode):
        self.encode = encode
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="arrow-encode")
        self.futures = {}

    def schedule(self, indices):
        indices = list(dict.fromkeys(indices))
        wanted = set(indices)
        for index in list(self.futures):
            if index not in wanted:
                self.futures.pop(index).cancel()
        for index in indices:
            if index not in self.futures:
                self.futures[index] = self.pool.submit(self.encode, index)

    def get(self, index):
        future = self.futures.pop(index, None)
        if future is None:
            future = self.pool.submit(self.encode, index)
        return future.result()

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.futures.clear()
