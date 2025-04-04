#!/usr/bin/env python3
"""
AI Visualizer Python Interface - Interactive CLI
A user-friendly command-line interface for working with neural network models
trained on aiviz.kasperlarsen.tech
"""

import curses
import os
import sys
import io
import time
from contextlib import redirect_stdout
from neural_network import NeuralNetwork

# Global variables
model = NeuralNetwork()
loaded_model_path = None
prediction_precision = 2  # Default precision for predictions
prediction_debug = False  # Default debug setting

def create_menu(menu_items, stdscr, title="Menu Options", current_model=None):
    """Create a menu using curses and return the selected item index"""
    stdscr.clear()
    height, width = stdscr.getmaxyx()
    
    # Draw header
    header_text = title
    stdscr.attron(curses.color_pair(1))
    stdscr.addstr(0, 0, " " * width)
    stdscr.addstr(0, (width - len(header_text)) // 2, header_text)
    stdscr.attroff(curses.color_pair(1))
    
    # Show current model status if available
    if current_model:
        model_text = f"Current model: {os.path.basename(current_model)}"
        stdscr.addstr(2, 2, model_text)
    else:
        stdscr.addstr(2, 2, "No model loaded")
    
    # Set the initial position of the cursor
    current_row = 0
    
    while True:
        # Print the menu
        for idx, item in enumerate(menu_items):
            y = 4 + idx
            x = 4
            
            if idx == current_row:
                stdscr.attron(curses.color_pair(2))
                stdscr.addstr(y, x, f"> {item}")
                stdscr.attroff(curses.color_pair(2))
            else:
                stdscr.addstr(y, x, f"  {item}")
        
        # Help text at bottom
        help_text = "Use UP/DOWN arrows to navigate, ENTER to select, Q to quit"
        if height > 10:  # Only show if we have enough room
            stdscr.attron(curses.color_pair(3))
            stdscr.addstr(height-2, (width - len(help_text)) // 2, help_text)
            stdscr.attroff(curses.color_pair(3))
        
        stdscr.refresh()
        
        # Get user input
        key = stdscr.getch()
        
        if key == curses.KEY_UP and current_row > 0:
            current_row = current_row - 1
        elif key == curses.KEY_DOWN and current_row < len(menu_items) - 1:
            current_row = current_row + 1
        elif key == curses.KEY_ENTER or key in [10, 13]:  # Enter key
            return current_row
        elif key == ord('q') or key == ord('Q'):  # Q key to quit
            return -1
        
        # Also support numeric selection
        if key >= ord('1') and key <= ord(str(min(9, len(menu_items)))):
            return key - ord('1')

def get_input(stdscr, prompt, y_pos=None, x_pos=None):
    """Get input from the user using curses"""
    height, width = stdscr.getmaxyx()
    
    if y_pos is None:
        y_pos = height // 2
    if x_pos is None:
        x_pos = 2
    
    # Display the prompt
    stdscr.addstr(y_pos, x_pos, prompt)
    stdscr.addstr(y_pos + 1, x_pos, "> ")
    stdscr.refresh()
    
    # Setup for input
    curses.echo()
    curses.curs_set(1)  # Show the cursor
    
    # Create a window for input
    input_y = y_pos + 1
    input_x = x_pos + 2
    input_win = curses.newwin(1, width - input_x - 2, input_y, input_x)
    input_win.refresh()
    
    # Get the input
    user_input = ""
    while True:
        key = input_win.getch()
        
        if key == 10:  # Enter key
            break
        elif key == 27:  # Escape key
            user_input = ""
            break
        elif key == curses.KEY_BACKSPACE or key == 127:  # Backspace
            if user_input:
                user_input = user_input[:-1]
                input_win.clear()
                input_win.addstr(0, 0, user_input)
                input_win.refresh()
        else:
            user_input += chr(key)
    
    # Reset cursor and echo state
    curses.noecho()
    curses.curs_set(0)  # Hide the cursor
    
    return user_input

def show_message(stdscr, message, title=None, wait_for_key=True):
    """Display a message to the user"""
    stdscr.clear()
    height, width = stdscr.getmaxyx()
    
    # Draw header if title is provided
    if title:
        stdscr.attron(curses.color_pair(1))
        stdscr.addstr(0, 0, " " * width)
        stdscr.addstr(0, (width - len(title)) // 2, title)
        stdscr.attroff(curses.color_pair(1))
    
    # Split message into lines if it contains newlines
    message_lines = message.split('\n')
    start_y = 2 if title else 0
    
    # Display each line of the message
    for i, line in enumerate(message_lines):
        if start_y + i < height:
            stdscr.addstr(start_y + i, 2, line)
    
    # Prompt to continue if wait_for_key is True
    if wait_for_key:
        prompt = "Press any key to continue..."
        if start_y + len(message_lines) + 2 < height:
            stdscr.attron(curses.color_pair(3))
            stdscr.addstr(start_y + len(message_lines) + 2, 2, prompt)
            stdscr.attroff(curses.color_pair(3))
        
        stdscr.refresh()
        stdscr.getch()  # Wait for a key press
    else:
        stdscr.refresh()

def confirmation_dialog(stdscr, message):
    """Display a yes/no confirmation dialog"""
    stdscr.clear()
    height, width = stdscr.getmaxyx()
    
    # Display the message
    stdscr.addstr(height // 3, 2, message)
    
    # Add options
    options = "Press Y for Yes, N for No"
    stdscr.attron(curses.color_pair(3))
    stdscr.addstr(height // 3 + 2, 2, options)
    stdscr.attroff(curses.color_pair(3))
    
    stdscr.refresh()
    
    # Wait for Y or N
    while True:
        key = stdscr.getch()
        if key in [ord('y'), ord('Y')]:
            return True
        elif key in [ord('n'), ord('N')]:
            return False

def display_scrollable_content(stdscr, content_lines, title="Information"):
    """Display scrollable content with navigation controls"""
    height, width = stdscr.getmaxyx()
    
    # Calculate how many lines we can display at once (minus header and footer)
    page_size = height - 6
    total_pages = (len(content_lines) + page_size - 1) // page_size  # Ceiling division
    current_page = 0
    
    while True:
        stdscr.clear()
        
        # Draw header
        stdscr.attron(curses.color_pair(1))
        stdscr.addstr(0, 0, " " * width)
        header_text = f"{title} (Page {current_page + 1}/{total_pages})"
        stdscr.addstr(0, (width - len(header_text)) // 2, header_text)
        stdscr.attroff(curses.color_pair(1))
        
        # Calculate start and end indices for the current page
        start_idx = current_page * page_size
        end_idx = min(start_idx + page_size, len(content_lines))
        
        # Display the content for the current page
        for i, line in enumerate(content_lines[start_idx:end_idx], 2):
            # Ensure we don't try to write outside the screen
            if i < height - 2:
                # Truncate line if it's too long for the screen
                display_line = line[:width-3] + "..." if len(line) > width-3 else line
                stdscr.addstr(i, 2, display_line)
        
        # Display navigation instructions
        nav_text = "UP/DOWN: scroll, HOME/END: first/last page, Q: back to menu"
        stdscr.attron(curses.color_pair(3))
        if height > 4:
            stdscr.addstr(height-2, (width - len(nav_text)) // 2, nav_text)
        stdscr.attroff(curses.color_pair(3))
        
        stdscr.refresh()
        
        # Handle user input for navigation
        key = stdscr.getch()
        
        if key == ord('q') or key == ord('Q'):  # Q to quit
            break
        elif key == curses.KEY_UP:  # Up arrow to scroll up
            if current_page > 0:
                current_page -= 1
        elif key == curses.KEY_DOWN:  # Down arrow to scroll down
            if current_page < total_pages - 1:
                current_page += 1
        elif key == curses.KEY_PPAGE:  # Page Up
            current_page = max(0, current_page - 1)
        elif key == curses.KEY_NPAGE:  # Page Down
            current_page = min(total_pages - 1, current_page + 1)
        elif key == curses.KEY_HOME:  # Home key for first page
            current_page = 0
        elif key == curses.KEY_END:  # End key for last page
            current_page = total_pages - 1

def load_model_menu(stdscr):
    """Menu for loading a neural network model"""
    global model, loaded_model_path
    
    stdscr.clear()
    
    # Draw header
    height, width = stdscr.getmaxyx()
    header_text = "Load Neural Network Model"
    stdscr.attron(curses.color_pair(1))
    stdscr.addstr(0, 0, " " * width)
    stdscr.addstr(0, (width - len(header_text)) // 2, header_text)
    stdscr.attroff(curses.color_pair(1))
    
    # Get the file path from the user
    filepath = get_input(stdscr, "Enter the path to your model file (.txt):", 2, 2)
    
    if not filepath:
        show_message(stdscr, "No file path provided.", "Load Cancelled")
        return
    
    try:
        new_model = NeuralNetwork()
        new_model.load_model(filepath)
        
        # If successful, update the global model
        model = new_model
        loaded_model_path = filepath
        
        success_message = (
            f"Model successfully loaded from {os.path.basename(filepath)}\n\n"
            f"Network architecture: {' → '.join(str(size) for size in model.layer_sizes)}"
        )
        show_message(stdscr, success_message, "Success")
    except Exception as e:
        error_message = f"Error loading model: {str(e)}"
        show_message(stdscr, error_message, "Error")

def view_model_info_menu(stdscr):
    """Display detailed information about the currently loaded model"""
    global model, loaded_model_path
    
    if not loaded_model_path:
        show_message(stdscr, "No model is currently loaded. Please load a model first.", "Error")
        return
    
    # Get model information as text
    info_text = str(model)
    info_lines = info_text.split('\n')
    
    # Display the model information in a scrollable view
    display_scrollable_content(stdscr, info_lines, "Model Information")

def make_prediction_menu(stdscr):
    """Menu for making predictions with the model"""
    global model, loaded_model_path, prediction_precision, prediction_debug
    
    if not loaded_model_path:
        show_message(stdscr, "No model is currently loaded. Please load a model first.", "Error")
        return
    
    while True:
        stdscr.clear()
        height, width = stdscr.getmaxyx()
        
        # Draw header
        header_text = "Make a Prediction"
        stdscr.attron(curses.color_pair(1))
        stdscr.addstr(0, 0, " " * width)
        stdscr.addstr(0, (width - len(header_text)) // 2, header_text)
        stdscr.attroff(curses.color_pair(1))
        
        # Display model info and options
        input_size = model.layer_sizes[0]
        stdscr.addstr(2, 2, f"Current model: {os.path.basename(loaded_model_path)}")
        stdscr.addstr(3, 2, f"This model requires {input_size} input values.")
        
        # Menu options
        options = [
            "Enter input values and make prediction",
            f"Change precision (currently: {prediction_precision if prediction_precision is not None else 'None'})",
            f"Toggle debug mode (currently: {'ON' if prediction_debug else 'OFF'})",
            "Return to main menu"
        ]
        
        # Get user choice using the create_menu function
        choice = create_menu(options, stdscr, "Make a Prediction", loaded_model_path)
        
        if choice == -1 or choice == 3:  # Exit/Back option
            break
        elif choice == 0:  # Enter input values
            handle_prediction_input(stdscr)
        elif choice == 1:  # Change precision
            change_precision(stdscr)
        elif choice == 2:  # Toggle debug mode
            prediction_debug = not prediction_debug
            show_message(stdscr, f"Debug mode {'enabled' if prediction_debug else 'disabled'}")

def handle_prediction_input(stdscr):
    """Handle the input for prediction and display results"""
    global model, prediction_precision, prediction_debug
    
    # Get the number of inputs required
    input_size = model.layer_sizes[0]
    
    # Get input values from user
    input_str = get_input(stdscr, f"Enter {input_size} values separated by spaces (e.g., '0.5 1.0 0'):")
    
    if not input_str:
        show_message(stdscr, "No input provided.", "Cancelled")
        return
    
    try:
        # Parse input values
        input_values = [float(x) for x in input_str.strip().split()]
        
        if len(input_values) != input_size:
            show_message(stdscr, f"Error: Expected {input_size} inputs, got {len(input_values)}", "Input Error")
            return
        
        # Make prediction
        if prediction_debug:
            # Capture debug output
            output = io.StringIO()
            with redirect_stdout(output):
                result = model.predict(*input_values, debug=True, precision=prediction_precision)
            
            # Add result information at the top
            debug_text = output.getvalue()
            result_header = f"PREDICTION RESULT:\n\nInput: {input_values}\nOutput: {result}\n\n"
            debug_lines = (result_header + debug_text).split('\n')
            
            # Display debug output in scrollable view
            display_scrollable_content(stdscr, debug_lines, "Prediction Debug Output")
        else:
            # Just show the result
            result = model.predict(*input_values, precision=prediction_precision)
            show_message(stdscr, f"Input: {input_values}\n\nPrediction result: {result}", "Prediction Result")
            
    except Exception as e:
        show_message(stdscr, f"Error making prediction: {str(e)}", "Error")

def change_precision(stdscr):
    """Change the precision setting for predictions"""
    global prediction_precision
    
    stdscr.clear()
    height, width = stdscr.getmaxyx()
    
    # Draw header
    header_text = "Set Precision"
    stdscr.attron(curses.color_pair(1))
    stdscr.addstr(0, 0, " " * width)
    stdscr.addstr(0, (width - len(header_text)) // 2, header_text)
    stdscr.attroff(curses.color_pair(1))
    
    # Display current setting and instructions
    current = "None" if prediction_precision is None else str(prediction_precision)
    stdscr.addstr(2, 2, f"Current precision: {current}")
    stdscr.addstr(4, 2, "Enter new precision (0-6) or 'none' for no rounding:")
    
    # Get user input
    precision_input = get_input(stdscr, "", 4, 2).lower()
    
    if not precision_input:
        return
    
    try:
        if precision_input == 'none':
            prediction_precision = None
            show_message(stdscr, "Precision set to None (no rounding)", "Success")
        else:
            precision = int(precision_input)
            if 0 <= precision <= 6:
                prediction_precision = precision
                show_message(stdscr, f"Precision set to {precision} decimal places", "Success")
            else:
                show_message(stdscr, "Precision must be between 0 and 6", "Error")
    except ValueError:
        show_message(stdscr, "Invalid input. Please enter a number between 0 and 6 or 'none'", "Error")

def run_xor_example_menu(stdscr):
    """Run the XOR gate example with the current model"""
    global model, loaded_model_path
    
    if not loaded_model_path:
        show_message(stdscr, "No model is currently loaded. Please load a model first.", "Error")
        return
    
    # Check if the model has the correct structure for XOR (2 inputs, 1 output)
    if model.layer_sizes[0] != 2 or model.layer_sizes[-1] != 1:
        error_msg = (
            f"Error: XOR example requires a model with 2 inputs and 1 output.\n"
            f"Current model has {model.layer_sizes[0]} inputs and {model.layer_sizes[-1]} outputs."
        )
        show_message(stdscr, error_msg, "Incompatible Model")
        return
    
    # Draw header
    stdscr.clear()
    height, width = stdscr.getmaxyx()
    header_text = "XOR Gate Example"
    stdscr.attron(curses.color_pair(1))
    stdscr.addstr(0, 0, " " * width)
    stdscr.addstr(0, (width - len(header_text)) // 2, header_text)
    stdscr.attroff(curses.color_pair(1))
    
    # Show computing message
    stdscr.addstr(2, 2, "Computing results...")
    stdscr.refresh()
    
    try:
        # Use fixed precision for clean display
        precision = 2
        
        # Test all 4 XOR combinations
        result_0_0 = model.predict(0, 0, precision=precision)[0]
        result_0_1 = model.predict(0, 1, precision=precision)[0]
        result_1_0 = model.predict(1, 0, precision=precision)[0]
        result_1_1 = model.predict(1, 1, precision=precision)[0]
        
        # Format results
        results = [
            "XOR Logic Gate Test Results:",
            "---------------------------",
            f"0 XOR 0 = {result_0_0}",
            f"0 XOR 1 = {result_0_1}",
            f"1 XOR 0 = {result_1_0}",
            f"1 XOR 1 = {result_1_1}",
            "",
            "Expected XOR outputs:",
            "-------------------",
            "0 XOR 0 = 0",
            "0 XOR 1 = 1",
            "1 XOR 0 = 1",
            "1 XOR 1 = 0",
            "",
            "Accuracy assessment:",
            "------------------",
            f"Output 0 ≈ {result_0_0:.2f} (should be close to 0)",
            f"Output 1 ≈ {result_0_1:.2f} (should be close to 1)",
            f"Output 2 ≈ {result_1_0:.2f} (should be close to 1)",
            f"Output 3 ≈ {result_1_1:.2f} (should be close to 0)"
        ]
        
        # Display results in a scrollable view
        display_scrollable_content(stdscr, results, "XOR Gate Example")
        
    except Exception as e:
        show_message(stdscr, f"Error running XOR example: {str(e)}", "Error")

def main(stdscr):
    """Main function that sets up the curses interface and menu loop"""
    global prediction_precision, prediction_debug
    
    # Setup terminal
    curses.curs_set(0)  # Hide cursor
    curses.start_color()
    curses.use_default_colors()
    
    # Define color pairs
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)    # Header
    curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_GREEN)   # Selected menu item
    curses.init_pair(3, curses.COLOR_YELLOW, -1)                  # Help text
    curses.init_pair(4, curses.COLOR_RED, -1)                     # Error messages
    
    # Main menu options
    menu_items = [
        "Load a model",
        "View model information",
        "Make a prediction",
        "Run example (XOR gate)",
        "Exit"
    ]
    
    # Set default values for prediction settings
    prediction_precision = 2
    prediction_debug = False
    
    # Main program loop
    while True:
        # Display the main menu and get user selection
        selected = create_menu(menu_items, stdscr, "AI Visualizer - Interactive CLI", loaded_model_path)
        
        if selected == -1 or selected == 4:  # Exit option or Q key
            break
        elif selected == 0:  # Load a model
            load_model_menu(stdscr)
        elif selected == 1:  # View model information
            view_model_info_menu(stdscr)
        elif selected == 2:  # Make a prediction
            make_prediction_menu(stdscr)
        elif selected == 3:  # Run XOR example
            run_xor_example_menu(stdscr)
    
    # Farewell message
    stdscr.clear()
    stdscr.addstr(0, 0, "Thank you for using AI Visualizer Python Interface!")
    stdscr.refresh()
    time.sleep(1)

if __name__ == "__main__":
    try:
        # Start the curses application
        curses.wrapper(main)
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nThe program will now exit.")
