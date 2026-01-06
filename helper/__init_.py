```python
# This is a simple calculator class that performs basic arithmetic operations.
class Calculator:
    """
    A class used to perform basic arithmetic operations.

    Attributes:
    ----------
    num1 : float
        The first number.
    num2 : float
        The second number.

    Methods:
    -------
    add()
        Returns the sum of num1 and num2.
    subtract()
        Returns the difference of num1 and num2.
    multiply()
        Returns the product of num1 and num2.
    divide()
        Returns the quotient of num1 and num2.
    """

    def __init__(self, num1, num2):
        """
        Initializes the Calculator class.

        Parameters:
        ----------
        num1 : float
            The first number.
        num2 : float
            The second number.

        Notes:
        -----
        The Calculator class assumes that num1 and num2 are valid numbers.
        """
        self.num1 = num1
        self.num2 = num2

    def add(self):
        """
        Returns the sum of num1 and num2.

        Returns:
        -------
        float
            The sum of num1 and num2.

        Examples:
        --------
        >>> calculator = Calculator(10, 2)
        >>> calculator.add()
        12
        """
        return self.num1 + self.num2

    def subtract(self):
        """
        Returns the difference of num1 and num2.

        Returns:
        -------
        float
            The difference of num1 and num2.

        Examples:
        --------
        >>> calculator = Calculator(10, 2)
        >>> calculator.subtract()
        8
        """
        return self.num1 - self.num2

    def multiply(self):
        """
        Returns the product of num1 and num2.

        Returns:
        -------
        float
            The product of num1 and num2.

        Examples:
        --------
        >>> calculator = Calculator(10, 2)
        >>> calculator.multiply()
        20
        """
        return self.num1 * self.num2

    def divide(self):
        """
        Returns the quotient of num1 and num2.

        Returns:
        -------
        float
            The quotient of num1 and num2.

        Raises:
        ------
        ZeroDivisionError
            If num2 is zero.

        Examples:
        --------
        >>> calculator = Calculator(10, 2)
        >>> calculator.divide()
        5.0
        """
        if self.num2 == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        return self.num1 / self.num2


# Example usage:
"""
This example demonstrates how to use the Calculator class to perform basic arithmetic operations.

1. Create an instance of the Calculator class with two numbers.
2. Use the add method to calculate the sum of the two numbers.
3. Use the subtract method to calculate the difference of the two numbers.
4. Use the multiply method to calculate the product of the two numbers.
5. Use the divide method to calculate the quotient of the two numbers.
"""
calculator = Calculator(10, 2)
print(calculator.add())  # Output: 12
print(calculator.subtract())  # Output: 8
print(calculator.multiply())  # Output: 20
print(calculator.divide())  # Output: 5.0
```