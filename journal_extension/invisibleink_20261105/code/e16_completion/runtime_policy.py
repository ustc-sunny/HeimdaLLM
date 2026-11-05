"""Do not terminate a healthy admitted training task for co-tenancy alone."""

def resource_stop_reason(state, low_since, now):
    if state['free_mib'] >= 1024:
        return None, None
    if low_since is None:
        low_since = now
    if now-low_since >= 60:
        return 'GPU0 free memory stayed below1024MiB for60seconds', low_since
    return None, low_since
