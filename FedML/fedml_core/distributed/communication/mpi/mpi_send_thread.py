import logging
import queue
import threading

from ..message import Message


class MPISendThread(threading.Thread):
    def __init__(self, comm, rank, size, name, q):
        super(MPISendThread, self).__init__()
        self._stop_event = threading.Event()
        self.comm = comm
        self.rank = rank
        self.size = size
        self.name = name
        self.q = q
        # A fatal exception in the manager's dispatch loop must be allowed to
        # terminate the MPI rank instead of hanging forever while Python joins
        # an idle transport thread.
        self.daemon = True

    def run(self):
        logging.debug("Starting " + self.name + ". Process ID = " + str(self.rank))
        # Keep sending messages that were queued before shutdown.  In
        # particular, a FedSGD server queues STOP messages immediately before
        # asking the communication manager to stop.
        while not self.stopped() or not self.q.empty():
            try:
                msg = self.q.get(timeout=0.05)
            except queue.Empty:
                continue

            try:
                dest_id = msg.get(Message.MSG_ARG_KEY_RECEIVER)
                msg_type = msg.get(Message.MSG_ARG_KEY_TYPE)
                logging.info(
                    "MPI send begin rank=%s type=%s dest=%s", self.rank,
                    msg_type, dest_id,
                )
                self.comm.send(msg.to_string(), dest=dest_id)
                logging.info(
                    "MPI send done rank=%s type=%s dest=%s", self.rank,
                    msg_type, dest_id,
                )
            except Exception:
                logging.exception("MPI send failed on rank %s", self.rank)
            finally:
                self.q.task_done()

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
