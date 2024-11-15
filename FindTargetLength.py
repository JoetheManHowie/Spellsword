from itertools import combinations_with_replacement
from collections import Counter

def analyze_length_combinations(target_sum, min_len=2, max_len=7):
    """Analyze combinations of integers that sum to target_sum with required lengths."""
    count = 0
    required_lengths = set(range(min_len, max_len + 1))
    max_possible_words = target_sum // min_len
    min_words_found = float('inf')
    max_words_found = 0
    
    def is_valid_combo(combo):
        return all(length in combo for length in required_lengths)
    
    for n_words in range(len(required_lengths), max_possible_words + 1):
        for combo in combinations_with_replacement(range(min_len, max_len + 1), n_words):
            if sum(combo) == target_sum and is_valid_combo(combo):
                count += 1
                min_words_found = min(min_words_found, len(combo))
                max_words_found = max(max_words_found, len(combo))
    
    return count, min_words_found if count > 0 else None, max_words_found if count > 0 else None

def find_viable_target_lengths(min_target=35, max_target=50, min_combinations=24):
    """Find target lengths that yield at least the minimum number of combinations."""
    results = []
    
    print(f"Searching for target lengths with {min_combinations}+ combinations...")
    print(f"{'Target Length':^12} | {'Combinations':^12} | {'Word Count Range':^15}")
    print("-" * 44)
    
    for target in range(min_target, max_target + 1):
        count, min_words, max_words = analyze_length_combinations(target)
        word_range = f"{min_words}-{max_words}" if count > 0 else "N/A"
        print(f"{target:^12} | {count:^12} | {word_range:^15}")
        
        if count >= min_combinations:
            results.append((target, count, min_words, max_words))
    
    return results

if __name__ == "__main__":
    viable_targets = find_viable_target_lengths(min_target=30, max_target=60)
    
    if viable_targets:
        print("\nViable target lengths:")
        for target, count, min_words, max_words in viable_targets:
            print(f"Target length {target}: {count} combinations (uses {min_words}-{max_words} words)")
            
            # For the first viable target, show some example combinations
            if target == viable_targets[0][0]:
                print(f"\nExample combinations for length {target}:")
                example_count = 0
                for n_words in range(min_words, max_words + 1):
                    for combo in combinations_with_replacement(range(2, 8), n_words):
                        if sum(combo) == target and all(i in combo for i in range(2, 8)):
                            print(f"{len(combo)} words: {sorted(combo)}")
                            example_count += 1
                            if example_count >= 5:  # Show first 5 combinations as examples
                                break
                    if example_count >= 5:
                        break
        
        # Show the best option
        best_target = min(viable_targets, key=lambda x: abs(x[1] - 24))
        print(f"\nRecommended target length: {best_target[0]} (yields {best_target[1]} combinations, uses {best_target[2]}-{best_target[3]} words)")
    else:
        print("\nNo target lengths found that yield 24+ combinations")