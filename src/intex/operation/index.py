import abc


class Index(abc.ABC):
    @abc.abstractmethod
    def to_csv(self, path: str):
        raise NotImplementedError

    @abc.abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError
