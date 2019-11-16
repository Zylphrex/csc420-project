def at_least(n):
    def predicate(iterable):
        return len(iterable) >= n
    return predicate
