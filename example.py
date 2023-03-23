import sys
from colorama import Fore
from random import randint

sys.path.append("./build/")
import pyllama_cpu as llama

MODEL_PATH = "./models/7B/ggml-model-q4_0.bin"

def flush_output(token: str) -> None:
    """
    Prints the given token in green color and flushes the output stream.

    Args:
    token (str): The token to be printed.

    Returns:
    None
    """
    print(Fore.GREEN + token, end='', flush=True)

# Instantiate the model object
model = llama.Model(
    path=MODEL_PATH, 
    num_threads=10,
    n_ctx=512, 
    last_n_size=64, 
    seed=randint(0, 10e5)
)

# Print the welcome message
print(Fore.YELLOW + "--------------------------------------------")
print(Fore.YELLOW + "----LLaMa text completion running on CPU----")
print(Fore.YELLOW + "-------Type 'exit' to close the loop--------")
print(Fore.YELLOW + "--------------------------------------------")

# Start the main loop
while True:
    # Prompt the user for input
    print(Fore.RED + '\n[Prompt]: ' + Fore.WHITE, end='')
    user_input = input()

    # Check if the user wants to exit
    if user_input == "exit":
        break

    # Ingest the user input
    res = model.ingest(user_input)

    # Check if ingestion was successful
    if res is not True:
        break

    # Print the user input
    print(Fore.WHITE + user_input, end='')

    # Generate text using the model
    res = model.generate(
        num_tokens=130,
        top_p=0.95,
        temp=0.8,
        repeat_penalty=1.0,
        streaming_fn=flush_output,
    )

    print("\n")
