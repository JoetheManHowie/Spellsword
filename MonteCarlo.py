import random
from collections import Counter
import csv
import multiprocessing as mp
import time
import os
from itertools import combinations

def initialize_bag():
    """Initialize the bag with the specified distribution of tiles."""
    bag = []
    # Common symbols
    for symbol in ['A', 'B', 'C', 'D']:
        bag.extend([symbol] * 16)
    # Special symbols
    for symbol in ['X', 'Y', 'Z']:
        bag.extend([symbol] * 8)
    # Rare symbols
    for symbol in ['J', 'K']:
        bag.extend([symbol] * 4)
    # Blank tiles
    bag.extend(['0'] * 4)
    return bag

def generate_words_from_tiles(tiles, blank_count):
    """Generate all possible words from the given tiles, considering one blank as wild."""
    # Remove blanks from tiles list
    real_tiles = [t for t in tiles if t != '0']
    words = set()
    
    # For each word length from 2 to min(7, number of tiles)
    for length in range(2, min(8, len(tiles) + 1)):
        # First, generate words without using blanks
        for combo in combinations(real_tiles, length):
            words.add(''.join(sorted(combo)))
        
        # If we have a blank, generate words using one blank
        if blank_count > 0:
            # For each possible length using one fewer real tile
            for base_length in range(max(1, length - 1), length):
                for base_combo in combinations(real_tiles, base_length):
                    # Get the unique set of letters we could add with the blank
                    possible_letters = set(['A', 'B', 'C', 'D', 'X', 'Y', 'Z', 'J', 'K'])
                    # Add each possible letter with the blank
                    for letter in possible_letters:
                        new_word = ''.join(sorted(base_combo + (letter,)))
                        if len(new_word) >= 2:
                            words.add(new_word)
    
    return words

def run_batch_simulation(batch_args):
    """Run a batch of simulations and return the counts."""
    batch_size, process_num, tiles_to_draw = batch_args
    word_counts = Counter()
    process = mp.current_process()
    
    for i in range(batch_size):
        if i % 1000 == 0:
            print(f"Process {process.name}: {i}/{batch_size} simulations complete")
            
        # Draw tiles
        bag = initialize_bag()
        drawn_tiles = random.sample(bag, tiles_to_draw)
        
        # Count blanks
        blank_count = sum(1 for t in drawn_tiles if t == '0')
        
        # Generate and count all possible words from these tiles
        possible_words = generate_words_from_tiles(drawn_tiles, blank_count)
        word_counts.update(possible_words)
    
    return word_counts

def run_parallel_simulation(num_simulations=100000, tiles_to_draw=7, num_processes=None):
    """Run the Monte Carlo simulation using parallel processing."""
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # Split simulations into batches
    batch_size = num_simulations // num_processes
    remaining = num_simulations % num_processes
    
    # Prepare batch arguments
    batch_args = []
    for i in range(num_processes):
        size = batch_size + (remaining if i == 0 else 0)
        batch_args.append((size, i, tiles_to_draw))
    
    # Use 'spawn' method for Windows compatibility
    ctx = mp.get_context('spawn')
    
    start_time = time.time()
    print("Starting parallel processing...")
    
    # Create a pool of worker processes
    with ctx.Pool(processes=num_processes) as pool:
        results = pool.map(run_batch_simulation, batch_args)
    
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
    
    # Combine results
    final_counts = Counter()
    for count_dict in results:
        final_counts.update(count_dict)
    
    return final_counts

def save_results(word_counts, filename='tile_probabilities.csv'):
    """Save the results to a CSV file, sorted by word length and then alphabetically."""
    sorted_results = sorted(word_counts.items(), key=lambda x: (len(x[0]), x[0]))
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Spell', 'Count', 'Probability'])
        total_sims = 100000  # We know this is our simulation count
        for word, count in sorted_results:
            prob = count / total_sims * 100
            writer.writerow([word, count, f"{prob:.2f}%"])

def main():
    num_cores = mp.cpu_count()
    print(f"Starting optimized Monte Carlo simulation using {num_cores} CPU cores...")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    word_counts = run_parallel_simulation(num_processes=num_cores)
    save_results(word_counts)
    print("Simulation complete! Results saved to tile_probabilities.csv")
    
    # Print some sample results
    print("\nSample of results (top 10 most common):")
    total_sims = 100000  # We know this is our simulation count
    for word, count in sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))[:10]:
        print(f"{word}: {count} ({count/total_sims*100:.1f}%)")

if __name__ == "__main__":
    main()