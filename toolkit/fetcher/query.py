# Given a list of files filter is based on the meta data.

class Query:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            print('keyword argument: {} = {}'.format(k, v))

    def _parse_query(self, query):
        """Parse the query string to set filters."""
        queries = query.split(',')
        for i, q in enumerate(queries):
            q = q.split('=')
            queries[i] = (q[0], q[1])
        return

    @staticmethod
    def query_files(self):
        return
