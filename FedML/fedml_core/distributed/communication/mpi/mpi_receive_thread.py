import logging
import threading

from mpi4py import MPI

from ..message import Message


class MPIReceiveThread(threading.Thread):
    def __init__(self, comm, rank, size, name, q):
        super(MPIReceiveThread, self).__init__()
        self._stop_event = threading.Event()
        self.comm = comm
        self.rank = rank
        self.size = size
        self.name = name
        self.q = q
        self.daemon = True

    def run(self):
        logging.debug("Starting Thread:" + self.name + ". Process ID = " + str(self.rank))
        # A blocking recv cannot observe the stop event.  Probe with a short
        # event wait instead so the owning manager can join this thread during
        # an orderly MPI shutdown.
        while not self.stopped():
            try:
                if self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG):
                    msg_str = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
                    msg = Message()
                    msg.init(msg_str)
                    logging.info(
                        "MPI receive rank=%s type=%s source=%s", self.rank,
                        msg.get(Message.MSG_ARG_KEY_TYPE),
                        msg.get(Message.MSG_ARG_KEY_SENDER),
                    )
                    self.q.put(msg)
                else:
                    self._stop_event.wait(0.01)
            except Exception:
                if not self.stopped():
                    logging.exception("MPI receive failed on rank %s", self.rank)
                    self._stop_event.wait(0.05)

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def get_id(self):
        # returns id of the respective thread
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

    def raise_exception(self):
        """Backward-compatible alias for the old unsafe thread shutdown."""
        self.stop()
