"""Port of src/MinHeap2.java -- binary min-heap implementation.

The heap is 1-indexed, exactly like the Java version: ``heap[0]`` is
unused (``None``) and valid elements occupy ``heap[1..heap_size]``.
"""


class MinHeap2:
    def __init__(self, arr, length):
        """``arr`` is a list of MinHeapElement, 1-indexed like the Java
        array (arr[0] is ignored/overwritten), of length >= length+1."""
        self.iconverter = [0] * length

        self.heap = arr
        self.heap[0] = None
        self.heap_size = length

        for i in range(1, self.heap_size + 1):
            self.iconverter[self.heap[i].index] = i

        self.build_min_heap()

    def parent(self, index):
        return index // 2

    def left(self, index):
        return 2 * index

    def right(self, index):
        return 2 * index + 1

    def min_heapify(self, index):
        temp = self._min_heapify_step(index)

        while temp != -1:
            temp = self._min_heapify_step(temp)

    def _min_heapify_step(self, index):
        heap = self.heap
        heap_size = self.heap_size

        l = 2 * index
        r = 2 * index + 1

        if l <= heap_size and heap[l].key < heap[index].key:
            smallest = l
        else:
            smallest = index

        if r <= heap_size and heap[r].key < heap[smallest].key:
            smallest = r

        if smallest != index:
            el1 = heap[index]
            el2 = heap[smallest]

            heap[index] = el2
            heap[smallest] = el1

            self.iconverter[el2.index] = index
            self.iconverter[el1.index] = smallest

            return smallest

        return -1

    def build_min_heap(self):
        for i in range(self.heap_size // 2, 0, -1):
            self.min_heapify(i)

    def extract_min(self):
        if self.heap_size < 1:
            print("WARNING in MinHeap2: min-heap is empty! Returning null...")
            return None

        heap = self.heap
        min_el = heap[1]

        self.iconverter[heap[1].index] = -1
        heap[1] = heap[self.heap_size]
        self.heap_size -= 1
        self.iconverter[heap[1].index] = 1

        self.min_heapify(1)

        return min_el

    def decrease_key(self, index_in_points, key):
        heap = self.heap
        index = self.iconverter[index_in_points]

        if key >= heap[index].key:
            print("WARNING in MinHeap2: new key is bigger than the current key! Ignoring...")
            return

        heap[index].key = key

        i = index
        parent_i = index // 2
        while i > 1 and heap[parent_i].key > heap[i].key:
            el1 = heap[i]
            el2 = heap[parent_i]
            heap[i] = el2
            heap[parent_i] = el1

            self.iconverter[el2.index] = i
            self.iconverter[el1.index] = parent_i

            i = parent_i
            parent_i = i // 2
