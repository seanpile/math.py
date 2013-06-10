#!/usr/bin/env python
# coding=utf-8
#
#  Copyright (c) 2013 Sean Dawson. All rights reserved.
#
#  This file is released under the MIT License:
#  http://www.opensource.org/licenses/mit-license.php

def dot_product(row, column):
    """Dot product between two vectors.
    >>> dot_product([1,2,3], [4,5,6])
    32
    >>> dot_product([1,2], [10, 30])
    70
    """
    return reduce(lambda x, y: x + y, [x * y for x, y in zip(row, column)])

class Matrix(object):
    """A Matrix.  Provides matrix operations

    >>> x = Matrix(rows=2, columns=3, data = [1, 2, 3, 4, 5, 6])
    >>> x.size()
    (2, 3)
    >>> x.row(1)
    [1, 2, 3]
    >>> x.row(2)
    [4, 5, 6]
    """

    def __init__(self, rows = 0, columns = 0, data = []):
        self._rows = rows
        self._columns = columns
        self._rank = -1

        if data:
            self.data = data[:]
        else:
            self.data = [0] * rows * columns

    @property
    def columns(self):
        return self._columns

    @property
    def rows(self):
        return self._rows

    def size(self):
        return (self.rows, self.columns)

    def row(self, i):
        return self.data[(self.columns * (i - 1)):(self.columns * i)]

    def column(self, i):
        """Returns the column at position i

        >>> m = Matrix(2, 3, data = [1, 2, 3, 4, 5, 6])
        >>> m.column(1)
        [1, 4]
        """
        return [self.data[ self.columns * row + (i - 1)] for row in range(self.rows)]

    def entry(self, i, j):
        """Entry; returns the ith-jth entry of this matrix (1 based)

        >>> x = Matrix(2,3, data = [1,2,3,4,5,6])
        >>> x.entry(1,3)
        3
        >>> x.entry(2,2)
        5
        """
        return self.data[self.columns * (i - 1) + j - 1]

    def transpose(self):
        """Returns a new matrix that is the transpose of self

        >>> m = Matrix(2,3, data=[1,2,3,4,5,6])
        >>> t = m.transpose()
        >>> t.size()
        (3, 2)
        >>> t.row(1)
        [1, 4]
        >>> t.column(1)
        [1, 2, 3]
        """
        transposed_data = []
        for i in range(1, self.columns + 1):
            transposed_data.extend(self.column(i))

        return Matrix(rows = self.columns, columns = self.rows, data = transposed_data)

    def __add__(self, other):
        """Matrix addition.  Sizes must be the same

        >>> x = Matrix(2,2, data=[1,2,3,4])
        >>> y = Matrix(2,2, data=[5,6,7,8])
        >>> z = x + y
        >>> z.size()
        (2, 2)
        >>> z.row(1)
        [6, 8]
        >>> z.row(2)
        [10, 12]
        """
        if not issubclass(type(other), Matrix):
            raise TypeError(type(other))

        if self.rows != other.rows or self.columns != other.columns:
            raise ValueError("Sizes should be equivalent")

        result = [ x + y for x, y in zip(self.data, other.data)]
        return Matrix(self.rows, self.columns, data = result)

    def __mul__(self, other):
        """Matrix multiplication.  If self.size() = (m, n), then other.size()[0] == n

        >>> x = Matrix(2,3, data=[1,2,3,4,5,6])
        >>> y = Matrix(3,1, data=[7, 8, 9])
        >>> z = x * y
        >>> z.size()
        (2, 1)
        >>> z.row(1)
        [50]
        >>> z.row(2)
        [122]
        >>> z = 10 * x
        >>> z.size()
        (2, 3)
        >>> z.row(1)
        [10, 20, 30]
        """

        # Scalar multiplication
        if isinstance(other, (int, long, float, complex)):
            return Matrix(self.rows, self.columns, [other * x for x in self.data])

        if not issubclass(type(other), Matrix):
            raise TypeError(type(other))

        if self.columns != other.rows:
            raise ValueError("Undefined multiplication for these matrices")

        result = []
        for i in range(1, self.rows + 1):
            row = self.row(i)
            result.extend([dot_product(row, other.column(j)) for j in range(1, other.columns + 1)])

        return Matrix(self.rows, other.columns, data = result)

    __rmul__ = __mul__

    def __eq__(self, other):
        """Matrix equality.

        >>> m = Matrix(2,3, data=[1,2,3,4,5,6])
        >>> n = Matrix(2,3, data=[1,2,3,4,5,6])
        >>> m == n
        True
        >>> m = Matrix(2,3, data=[1,2,3,4,5,6])
        >>> n = Matrix(3,2, data=[1,2,3,4,5,6])
        >>> m == n
        False
        >>> m = Matrix(2,3, data=[1,2,3,4,5,6])
        >>> n = Matrix(2,3, data=[1,5,3,4,2,1])
        >>> m == n
        False
        """
        if not issubclass(type(other), Matrix):
            return False

        if self.rows != other.rows or self.columns != other.columns:
            return False

        return self.data == other.data


    def is_identity(self):
        """Returns true if this is the identity matrix

        >>> x = Matrix(2,2, data = [1,0,0,1])
        >>> x.is_identity()
        True
        >>> x = Matrix(2,2, data = [1,3,0,1])
        >>> x.is_identity()
        False
        """

        if self.rows != self.columns:
            return False

        for i in range(self.rows):
            row = self.row(i + 1)
            for j in range(self.columns):
                if i == j and row[j] != 1:
                    return False

                if i != j and row[j] != 0:
                    return False

        return True;

    def is_inverse(self, other):
        """Returns true iff other is the inverse of self.  That is, if A = self, and B = other, then AB = I and BA = I.

        >>> A = Matrix(2,2, data = [0, 1, 1, 1])
        >>> B = Matrix(2,2, data = [-1, 1, 1, 0])
        >>> A.is_inverse(B)
        True
        >>> A = Matrix(2,2, data = [0, 1, 1, 1])
        >>> B = Matrix(2,2, data = [-10, 5, 1, 0])
        >>> A.is_inverse(B)
        False
        """
        return (self * other).is_identity() and (other * self).is_identity()

    def is_row_echelon(self):
        """Returns true iff this matrix is in row echelon format.

        >>> x = Matrix(2, 3, data=[1, 4, 5, 0, 0, 1])
        >>> x.is_row_echelon()
        True
        >>> x = Matrix(2, 3, data=[1, 4, 5, 1, 0, 1])
        >>> x.is_row_echelon()
        False
        """
        return self._is_row_echelon(False)

    def is_reduced_row_echelon(self):
        """Returns true iff this matrix is in row echelon format.

        >>> x = Matrix(2, 3, data=[1, 4, 5, 0, 0, 1])
        >>> x.is_reduced_row_echelon()
        False
        >>> x = Matrix(2, 3, data=[1, 4, 0, 0, 0, 1])
        >>> x.is_reduced_row_echelon()
        True
        >>> x = IdentityMatrix(10)
        >>> x.is_reduced_row_echelon()
        True
        >>> x.is_row_echelon()
        True
        """
        return self._is_row_echelon(True)


    def _is_row_echelon(self, check_reduced):

        found_zero_row = False
        last_leading_column = -1
        for i in range(1, self.rows + 1):

            row = self.row(i)
            is_zero_row = not reduce(lambda x, y: x or y, row)

            # Empty rows should appear at the bottom
            if (not is_zero_row) and found_zero_row:
                return False

            if is_zero_row:
                found_zero_row = True
                continue

            for j, value in enumerate(row):
                if value:
                    # Leading value should be 1
                    if value != 1:
                        return False

                    # Leading one's should always be down and to the right of the previous leading ones
                    if j <= last_leading_column:
                        return False

                    # Check if we are also in reduced row echelon form
                    if check_reduced and i > 1:
                        column = self.column(j + 1)
                        if reduce(lambda x, y: x or y, column[0:i-1]):
                            return False

                    last_leading_column = j
                    break

        return True

    def interchange(self, p, q):
        """Elementary row operation: interchange row p with row q.

        >>> x = Matrix(2,2, data = [1, 2, 3, 4])
        >>> x.interchange(1,2)
        >>> x.row(1)
        [3, 4]
        >>> x.row(2)
        [1, 2]
        """
        # Nothing to do if p == q
        if p == q:
            return

        tmp = self.data[self.columns * (p - 1):self.columns * p]
        self.data[self.columns * (p - 1):self.columns * p] = self.data[self.columns * (q - 1):self.columns * q]
        self.data[self.columns * (q - 1):self.columns * q] = tmp

    def multiply(self, p, c):
        """Elementary row operation: multiple row p by scalar c (c != 0)

        >>> x = Matrix(2, 2, data = [1,2,3,4])
        >>> x.multiply(2, 10)
        >>> x.row(2)
        [30, 40]
        >>> x.multiply(2, 0)
        Traceback (most recent call last):
        ValueError: Invalid constant == 0
        """
        if c == 0:
            raise ValueError("Invalid constant == 0")

        for x in range(self.columns * (p - 1), self.columns * p):
            self.data[x] *= c

    def add(self, p, q, k):
        """Elementary row operation: add k times row p to row q (p != q)

        >>> x = Matrix(2,2, data = [1,2,3,4])
        >>> x.add(1, 2, 5)
        >>> x.row(2)
        [8, 14]
        """
        if p == q:
            raise ValueError("Invalid target; p == q")

        for x in range(self.columns):
            self.data[self.columns * (q - 1) + x] += k * self.data[self.columns * (p - 1) + x]

    def to_reduced_row_echelon(self):
        """Apply the Gaussian Algorithm to produce a matrix in row echelon form.  Returns the resulting matrix and a list of
        the elementary operations applied.

        >>> x = Matrix(4, 6, data = [0, 0, 0, 2, 1, 9, 0, -2, -6, 2, 0, 2, 0, 2, 6, -2, 2, 0, 0, 3, 9, 2, 2, 19])
        >>> x.is_row_echelon()
        False
        >>> x.is_reduced_row_echelon()
        False
        >>> y, ops = x.to_reduced_row_echelon()
        >>> y.is_row_echelon()
        True
        >>> y.is_reduced_row_echelon()
        True
        """
        return self._to_row_echelon(fully_reduce = True)

    def to_row_echelon(self):
        """Apply the Gaussian Algorithm to produce a matrix in row echelon form.  Returns the resulting matrix and a list of
        the elementary operations applied.

        >>> x = Matrix(4, 6, data = [0, 0, 0, 2, 1, 9, 0, -2, -6, 2, 0, 2, 0, 2, 6, -2, 2, 0, 0, 3, 9, 2, 2, 19])
        >>> x.is_row_echelon()
        False
        >>> y, ops = x.to_row_echelon()
        >>> y.is_row_echelon()
        True
        """
        return self._to_row_echelon(fully_reduce = False)

    def _to_row_echelon(self, fully_reduce = False):

        row_idx = 1
        result = Matrix(self.rows, self.columns, data = self.data[:])
        operations = []

        for j in range(1, self.columns + 1):
            for i in range(row_idx, self.rows + 1):

                # non zero entry
                val = result.entry(i, j)
                if val:

                    # Interchange rows
                    if row_idx < i:
                        result.interchange(row_idx, i)
                        operations.append((result.interchange.__name__, row_idx, i))

                    # Simplify to leading 1
                    if val != 1:
                        result.multiply(row_idx, 1 / float(val))
                        operations.append((result.multiply.__name__, row_idx, 1 / float(val)))

                    # Reduce rows below
                    for k in range(max(row_idx + 1, i), self.rows + 1):
                        # Only care about reducing non-zero columns
                        multiple = result.entry(k, j)
                        if not multiple:
                            continue;

                        result.add(row_idx, k, -multiple)
                        operations.append((result.add.__name__, row_idx, k, -multiple))

                    row_idx += 1
                    break;

        # Do back substitution if we are fully reducing
        if fully_reduce:
            leading_row_idx = self.rows
            for j in range(self.columns, 0, -1):

                found_leading_one = False
                for i in range(leading_row_idx, 0, -1):
                    if (not found_leading_one) and result.entry(i, j) == 1:
                        found_leading_one = True
                        leading_row_idx = i
                    elif found_leading_one and result.entry(i, j):
                        multiple = result.entry(i, j)
                        result.add(leading_row_idx, i, -multiple)
                        operations.append((result.add.__name__, leading_row_idx, i, -multiple))

        return (result, operations)

    def invert(self):
        """Matrix Inversion; attempt to find the inverted matrix, or 0 if self is not invertible.

        >>> A = Matrix(3, 3, data = [2, 7, 1, 1, 4, -1, 1, 3, 0])
        >>> B = A.invert()
        >>> B.row(1)
        [-1.5, -1.5, 5.5]
        >>> B.row(2)
        [0.5, 0.5, -1.5]
        >>> B.row(3)
        [0.5, -0.5, -0.5]
        >>> C = A * B
        >>> C.is_identity()
        True
        >>> C = B * A
        >>> C.is_identity()
        True
        """

        if self.rows != self.columns:
            raise ValueError("Matrix must be square to invert")

        A, operations = self.to_reduced_row_echelon()
        if not A.is_identity():
            return 0

        # If A was reduced to the identity matrix, then the same set of operations will take I to the inverse of A.
        #   [A I] -> [I A^(-1)]

        I = IdentityMatrix(size = self.rows)
        for operation in operations:
            func = I.__getattribute__(operation[0])
            args = operation[1:]
            func(*args)

        return I

    def adjoint(self):
        """Returns the adjoint of A; that is, the transpose of the coefficient matrix of A.

        >>> A = Matrix(3, 3, data = [1, 3, -2,  0, 1, 5,  -2, -6, 7])
        >>> adj = A.adjoint()
        >>> adj.row(1)
        [37, -9, 17]
        >>> adj.row(2)
        [-10, 3, -5]
        >>> adj.row(3)
        [2, 0, 1]
        """
        data = []
        for i in range(1, self.rows + 1):
            for j in range(1, self.columns + 1):
                data.append(self._cofactor(i, j))

        mat = Matrix(self.rows, self.columns, data)
        return mat.transpose()

    def _cofactor(self, i, j):
        # Splice out row i and column j
        data = [val for k, val in enumerate(self.data) if k / self.columns != i - 1 and k % self.columns != j - 1]
        return (-1)**(i + j) * Matrix(self.rows - 1, self.rows - 1, data).det()

    def trace(self):
        """Returns the sum of diagonal elements for a square matrix.  Denoted tr A

        >>> x = Matrix(2,2, data=[1,5,20,40])
        >>> x.trace()
        41
        >>> x = Matrix(1,1, data=[5])
        >>> x.trace()
        5
        >>> x = Matrix(1,2, data=[5,8])
        >>> x.trace()
        Traceback (most recent call last):
        ValueError: Trace undefined for non-square matrices
        """
        if self.rows != self.columns:
            raise ValueError("Trace undefined for non-square matrices")

        return reduce(lambda x,y: x + y, [self.data[self.columns * i + i] for i in range(self.rows)])

    def rank(self):
        """Matrix rank; returns the number of leading 1's for the row echelon form of this matrix

        >>> a = Matrix(2,2, data = [1,0,0,1])
        >>> a.rank()
        2
        >>> x = Matrix(4, 6, data = [0, 0, 0, 2, 1, 9, 0, -2, -6, 2, 0, 2, 0, 2, 6, -2, 2, 0, 0, 3, 9, 2, 2, 19])
        >>> x.rank()
        3
        """

        if self._rank >= 0:
            return self._rank

        reduced, operations = self.to_row_echelon()
        non_leading_rows = 0
        for i in range(self.rows, 0, -1):
            if not reduce(lambda x,y: x or y, reduced.row(i)):
                non_leading_rows += 1
            else:
                break

        self._rank = self.rows - non_leading_rows
        return self._rank

    def det(self):
        """Determinant (det A).  Returns the determinant of a square matrix using Laplace expansion

        >>> A = Matrix(1,1, data = [3])
        >>> A.det()
        3
        >>> A = Matrix(2,2, data = [3, 5, 2, 4])
        >>> A.det()
        2
        >>> A = Matrix(3,3, data = [3, 4, 5, 1, 7, 2, 9, 8, -6])
        >>> A.det()
        -353
        """

        if self.rows != self.columns:
            raise ValueError("Matrix must be square")

        if self.rows == 1:
            return self.row(1)[0]

        if self.rows == 2:
            return self.entry(1,1) * self.entry(2,2) - self.entry(1,2) * self.entry(2,1)

        det = 0
        row_to_expand = 1

        for i in range(1, self.columns + 1):
            det += self.entry(row_to_expand, i) * self._cofactor(row_to_expand, i)

        return det

    def __repr__(self):

        if self.rows == 0 or self.columns == 0:
            return "0"

        formatted = []
        max_column_sizes = [0] * self.columns

        for row in range(self.rows):
            formatted_row = []
            for i, val in enumerate(self.row(row + 1)):
                to_add = ""

                if type(val) == int:
                    to_add = str(val)
                elif type(val) == float:
                    if val.is_integer():
                        to_add = str(int(val))
                    else:
                        integer_ratio = val.as_integer_ratio()
                        to_add = str(integer_ratio[0]) + "/" + str(integer_ratio[1])

                formatted_row.append(to_add)
                max_column_sizes[i] = max(max_column_sizes[i], len(to_add))

            formatted.append(formatted_row)

        repr_str = ""

        # 1 column
        if self.rows == 1:
            repr_str += "[ " + "  ".join(formatted[0]) + " ]"
        else:
            num_rows = self.rows
            for m, row in enumerate(formatted):
                joined = "  ".join([val.ljust(max_column_sizes[n]) for n, val in enumerate(row)])
                if m == 0:
                    repr_str += "⎡ " + joined + " ⎤\n"
                elif m == num_rows - 1:
                    repr_str += "⎣ " + joined + " ⎦"
                else:
                    repr_str += "⎢ " + joined + " ⎥\n"

        return repr_str


class EmptyMatrix(Matrix):
    def __init__(self):
        Matrix.__init__(self, 0, 0)

class IdentityMatrix(Matrix):
    """An identity Matrix (necessarily square)

    >>> x = IdentityMatrix(size = 3)
    >>> x.size()
    (3, 3)
    >>> x.row(1)
    [1, 0, 0]
    >>> x.row(2)
    [0, 1, 0]
    >>> x.row(3)
    [0, 0, 1]
    """
    def __init__(self, size):
        Matrix.__init__(self, size, size)

        column = 0
        for i in range(size):
            self.data[size * i + column] = 1
            column += 1


if __name__ == "__main__":
    import doctest
    doctest.testmod()
