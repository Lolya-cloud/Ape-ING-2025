from cloudpathlib import CloudPath


def read_bytes(file_path: str) -> bytes:
    try:
        # option 1: google cloud storage
        if file_path.startswith("gs://"):
            with CloudPath(file_path).open("rb") as f:
                return f.read()

        # option 3: local file storage
        else:
            with open(file_path, "rb") as f:
                return f.read()

    except Exception as exc:
        print(exc)
        raise exc
