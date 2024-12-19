import sys
import os 
from src.logger import logging

def error_message_detail(error,error_detail:sys):
    """
    Extracts detailed information about an error, including the script name,
    line number, and error message.

    Args:
        error (Exception): The error/exception object that was raised.
        error_detail (module): The sys module, used to extract exception details.

    Returns:
        str: A formatted error message containing script name, line number, and error description.
    """
    # Extract traceback object for the error
    _,_,exc_tb=error_detail.exc_info()

    # Retrieve the filename where the error occurred
    file_name=exc_tb.tb_frame.f_code.co_filename

    # Format the error message with script name, line number, and error message
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
    file_name,exc_tb.tb_lineno,str(error))

    return error_message


class CustomeException(Exception):
    """
    Custom exception class for handling and logging exceptions with detailed error messages.
    Extends the base Exception class.

    """
    def __init__(self, error_message, error_detail:sys):
         """
         Initializes the CustomException with a detailed error message.

         Args:
            error_message (str): The original error message.
            error_detail (module): The sys module to extract exception details.
         """
         #Call the base class constructor
         super().__init__(error_message)
         #Generate a detailed error message using the helper function
         self.error_message=error_message_detail(error_message, errerror_detail = error_detail)

    def __str__(self):
         """
        Returns the detailed error message when the exception is converted to a string.

        Returns:
            str: The detailed error message.
        """
         return self.error_message