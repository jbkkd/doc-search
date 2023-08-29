import argparse
import os

from dotenv import load_dotenv

load_dotenv()

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

PERSIST_DIR = os.getenv("PERSIST_DIR", "~/.doc-search/storage")


def search(query):
    try:
        index = rebuild_index_from_disk()
    except Exception as e:
        index = build_new_index(e)

    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    print(response)


def build_new_index(directory):
    print(f"Creating new index of {directory}")
    documents = SimpleDirectoryReader(input_dir=directory, recursive=True).load_data()
    index = VectorStoreIndex(documents)
    index.storage_context.persist(persist_dir="~/.doc-search/storage")
    return index


def rebuild_index_from_disk() -> VectorStoreIndex:
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    # load index
    index = load_index_from_storage(storage_context)
    print(index)
    print(type(index))
    return index


def convert_argparse_directory_to_absolute_path(directory):
    return os.path.abspath(directory)


if __name__ == "__main__":
    # Using argparse, either build a new index or search the index.
    # Build is a parameter that takes a directory, whereas for search you don't need any
    # parameters - you just pass the query as a string to the positional argument.
    parser = argparse.ArgumentParser(description="Llama Index")
    parser.add_argument(
        "--build",
        dest="build_directory",
        action="store",
        type=convert_argparse_directory_to_absolute_path,
        help="Build a new index from the given directory",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="",
        help="Query the index for the given string",
    )

    args = parser.parse_args()
    if args.build_directory:
        build_new_index(args.build_directory)
    else:
        search(args.query)
