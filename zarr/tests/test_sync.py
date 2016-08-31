# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from tempfile import mkdtemp
import atexit
import json
import shutil


from nose.tools import eq_ as eq


from zarr.tests.test_attrs import TestAttributes
from zarr.tests.test_core import TestArray
from zarr.sync import ThreadSynchronizer, ProcessSynchronizer
from zarr.core import Array
from zarr.attrs import Attributes
from zarr.storage import init_array
from zarr.compat import PY2


class TestThreadSynchronizedAttributes(TestAttributes):

    def init_attributes(self, store, read_only=False):
        key = 'attrs'
        store[key] = json.dumps(dict()).encode('ascii')
        synchronizer = ThreadSynchronizer()
        return Attributes(store, synchronizer=synchronizer, key=key,
                          read_only=read_only)


class TestProcessSynchronizedAttributes(TestAttributes):

    def init_attributes(self, store, read_only=False):
        key = 'attrs'
        store[key] = json.dumps(dict()).encode('ascii')
        sync_path = mkdtemp()
        atexit.register(shutil.rmtree, sync_path)
        synchronizer = ProcessSynchronizer(sync_path)
        return Attributes(store, synchronizer=synchronizer, key=key,
                          read_only=read_only)


class TestThreadSynchronizedArray(TestArray):

    def create_array(self, store=None, path=None, read_only=False,
                     chunk_store=None, **kwargs):
        if store is None:
            store = dict()
        init_array(store, path=path, chunk_store=chunk_store, **kwargs)
        return Array(store, path=path, synchronizer=ThreadSynchronizer(),
                     read_only=read_only, chunk_store=chunk_store)

    def test_repr(self):
        if not PY2:

            z = self.create_array(shape=100, chunks=10, dtype='f4',
                                  compression='zlib', compression_opts=1)
            # flake8: noqa
            expect = """zarr.core.Array((100,), float32, chunks=(10,), order=C)
  compression: zlib; compression_opts: 1
  nbytes: 400; nbytes_stored: 210; ratio: 1.9; initialized: 0/10
  store: builtins.dict
  synchronizer: zarr.sync.ThreadSynchronizer
"""
            actual = repr(z)
            for l1, l2 in zip(expect.split('\n'), actual.split('\n')):
                eq(l1, l2)


class TestProcessSynchronizedArray(TestArray):

    def create_array(self, store=None, path=None, read_only=False,
                     chunk_store=None, **kwargs):
        if store is None:
            store = dict()
        init_array(store, path=path, chunk_store=chunk_store, **kwargs)
        sync_path = mkdtemp()
        atexit.register(shutil.rmtree, sync_path)
        synchronizer = ProcessSynchronizer(sync_path)
        return Array(store, path=path, synchronizer=synchronizer,
                     read_only=read_only, chunk_store=chunk_store)

    def test_repr(self):
        if not PY2:

            z = self.create_array(shape=100, chunks=10, dtype='f4',
                                  compression='zlib', compression_opts=1)
            # flake8: noqa
            expect = """zarr.core.Array((100,), float32, chunks=(10,), order=C)
  compression: zlib; compression_opts: 1
  nbytes: 400; nbytes_stored: 210; ratio: 1.9; initialized: 0/10
  store: builtins.dict
  synchronizer: zarr.sync.ProcessSynchronizer
"""
            actual = repr(z)
            for l1, l2 in zip(expect.split('\n'), actual.split('\n')):
                eq(l1, l2)


# TODO group tests with synchronizer