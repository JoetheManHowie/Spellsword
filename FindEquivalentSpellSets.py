import pandas as pd
from collections import Counter, defaultdict
from itertools import combinations_with_replacement
import random
import time
from datetime import datetime
import sys
from typing import List, Dict, Set, Tuple

def get_canonical_word(word: List[str]) -> str:
    """Convert a list of letters to a canonical (sorted) word string."""
    return ''.join(sorted(word))

class SpellSet:
    def __init__(self, lengths: List[int], letters: Counter, spell_data: pd.DataFrame):
        self.lengths = lengths
        self.available_letters = letters.copy()
        self.spell_data = spell_data
        self.words: List[List[str]] = []
        
        # Cache valid words by length - Note: words in spell_data are already canonical
        self.valid_words_by_length = defaultdict(list)
        for _, row in spell_data.iterrows():
            spell = row['Spell']
            self.valid_words_by_length[len(spell)].append(spell)
        
        self.initialize_words()
    
    def initialize_words(self):
        """Create initial word slots and fill them with available letters."""
        letter_pool = []
        for letter, count in self.available_letters.items():
            letter_pool.extend([letter] * count)
        random.shuffle(letter_pool)
        
        current_pos = 0
        for length in self.lengths:
            word = letter_pool[current_pos:current_pos + length]
            self.words.append(word)
            current_pos += length
    
    def get_word_score(self, word: List[str]) -> float:
        """Get score for a word from the spell data."""
        word_str = get_canonical_word(word)  # Sort the letters before lookup
        if word_str not in self.valid_words_by_length[len(word)]:
            print(f"Invalid word found: '{word_str}' (from {''.join(word)})")
            return 0.0
        return self.spell_data.loc[self.spell_data['Spell'] == word_str, 'Score'].iloc[0]
    
    def get_total_score(self) -> float:
        """Get total score for the spell set."""
        return sum(self.get_word_score(word) for word in self.words)
    
    def try_swap(self, word1_idx: int, pos1: int, word2_idx: int, pos2: int) -> float:
        """Calculate the score change if we were to swap letters at the given positions."""
        word1 = self.words[word1_idx].copy()
        word2 = self.words[word2_idx].copy()
        
        # Perform the swap on copies
        word1[pos1], word2[pos2] = word2[pos2], word1[pos1]
        
        # Convert to canonical form for validity check
        word1_str = get_canonical_word(word1)
        word2_str = get_canonical_word(word2)
        
        # Debug print for invalid words
        if word1_str not in self.valid_words_by_length[len(word1)]:
            print(f"Would create invalid word: '{word1_str}' (from {''.join(word1)})")
        if word2_str not in self.valid_words_by_length[len(word2)]:
            print(f"Would create invalid word: '{word2_str}' (from {''.join(word2)})")
        
        old_score1 = self.get_word_score(self.words[word1_idx])
        old_score2 = self.get_word_score(self.words[word2_idx])
        new_score1 = self.get_word_score(word1)
        new_score2 = self.get_word_score(word2)
        
        if new_score1 == 0 or new_score2 == 0:
            return None
            
        return self.get_total_score() - (old_score1 + old_score2) + (new_score1 + new_score2)
    
    def perform_swap(self, word1_idx: int, pos1: int, word2_idx: int, pos2: int):
        """Perform the swap operation."""
        self.words[word1_idx][pos1], self.words[word2_idx][pos2] = \
            self.words[word2_idx][pos2], self.words[word1_idx][pos1]
    
    def optimize_score(self, target_score: float, tolerance: float = 1.0, 
                      max_attempts: int = 1000, progress=None) -> bool:
        """Try to optimize the spell set score through swaps."""
        current_score = self.get_total_score()
        best_score_diff = abs(current_score - target_score)
        best_state = [word.copy() for word in self.words]
        attempts = 0
        
        while best_score_diff > tolerance and attempts < max_attempts:
            attempts += 1
            
            need_increase = current_score < target_score
            potential_swaps = []
            
            # Try all possible swaps
            for i, word1 in enumerate(self.words):
                for j, word2 in enumerate(self.words[i+1:], i+1):
                    for pos1 in range(len(word1)):
                        for pos2 in range(len(word2)):
                            new_score = self.try_swap(i, pos1, j, pos2)
                            if new_score is not None:
                                score_diff = abs(new_score - target_score)
                                if (need_increase and new_score > current_score) or \
                                   (not need_increase and new_score < current_score):
                                    potential_swaps.append((i, pos1, j, pos2, score_diff))
            
            if not potential_swaps:
                break
                
            # Sort swaps by how close they get us to target score
            potential_swaps.sort(key=lambda x: x[4])
            
            # Perform the best swap
            word1_idx, pos1, word2_idx, pos2, score_diff = potential_swaps[0]
            self.perform_swap(word1_idx, pos1, word2_idx, pos2)
            current_score = self.get_total_score()
            
            if score_diff < best_score_diff:
                best_score_diff = score_diff
                best_state = [word.copy() for word in self.words]
            
            if progress:
                # Convert words to strings, showing both original and canonical forms for progress
                word_strings = []
                for word in self.words:
                    orig = ''.join(word)
                    can = get_canonical_word(word)
                    word_strings.append(f"{orig} ({can})")
                progress.update(current_score, best_score_diff, attempts, word_strings)
        
        if best_score_diff < abs(current_score - target_score):
            self.words = best_state
            
        return best_score_diff <= tolerance

    def get_words_as_strings(self) -> List[str]:
        """Convert word lists to strings, using canonical form for consistency."""
        return [get_canonical_word(word) for word in self.words]

class ProgressTracker:
    def __init__(self, spell_data, update_interval=1.0):
        self.start_time = time.time()
        self.last_update = self.start_time
        self.update_interval = update_interval
        self.solutions_found = 0
        self.combinations_tried = 0
        self.current_combo = None
        self.current_score = 0
        self.best_score_diff = float('inf')
        self.attempts = 0
        self.current_words = []
        self.spell_data = spell_data
    
    def update(self, current_score: float, best_score_diff: float, attempts: int, 
               current_words: List[str], force: bool = False):
        """Update progress information and display if enough time has passed."""
        current_time = time.time()
        self.current_score = current_score
        self.best_score_diff = best_score_diff
        self.attempts = attempts
        self.current_words = current_words
        
        if force or (current_time - self.last_update >= self.update_interval):
            self._print_progress()
            self.last_update = current_time
    
    def _print_progress(self):
        """Display current progress."""
        elapsed = time.time() - self.start_time
        
        sys.stdout.write("\033[H\033[J")  # Clear screen
        print(f"Progress Report - {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 50)
        print(f"Time elapsed: {elapsed:.1f}s")
        print(f"Combinations tried: {self.combinations_tried}")
        print(f"Solutions found: {self.solutions_found}")
        print(f"Current attempt: {self.attempts}")
        
        if self.current_combo:
            print(f"\nLength combination: {self.current_combo}")
        
        if self.current_words:
            print(f"\nTotal Score: {self.current_score:.3f}")
            print(f"Distance from target: {self.best_score_diff:.3f}")
            print("\nCurrent spell set:")
            for word in self.current_words:
                # The word string now contains both forms "orig (can)"
                try:
                    orig, can = word.split(" (")
                    can = can.rstrip(")")  # Remove the closing parenthesis
                    score = self.spell_data.loc[self.spell_data['Spell'] == can, 'Score'].iloc[0]
                    print(f"  {word}: {score:.3f}")
                except Exception as e:
                    print(f"  {word}: INVALID")
        
        print("-" * 50)
        sys.stdout.flush()

def find_valid_length_combinations(target_sum=43, min_len=2, max_len=7, min_words=8, max_words=10):
    """Find all combinations of integers that sum to target_sum with required lengths."""
    valid_combinations = []
    required_lengths = set(range(min_len, max_len + 1))
    
    def is_valid_combo(combo):
        return all(length in combo for length in required_lengths)
    
    for n_words in range(min_words, max_words + 1):
        for combo in combinations_with_replacement(range(min_len, max_len + 1), n_words):
            if sum(combo) == target_sum and is_valid_combo(combo):
                valid_combinations.append(list(combo))
    
    valid_combinations.sort(key=lambda x: -len(set(x)))
    return valid_combinations

def find_equivalent_spell_set(spell_data, target_length=43, target_score=211.887,
                            target_profile=Counter({'A': 8, 'B': 7, 'C': 7, 'D': 7,
                                                  'X': 4, 'Y': 3, 'Z': 3, 'J': 3, 'K': 1}),
                            tolerance=1.0):
    """Find spell sets using the swap optimization approach."""
    print("Finding valid length combinations...")
    length_combinations = find_valid_length_combinations(target_length)
    print(f"Found {len(length_combinations)} valid length combinations")
    
    progress = ProgressTracker(spell_data)
    valid_solutions = []
    
    for i, length_combo in enumerate(length_combinations):
        progress.combinations_tried = i + 1
        progress.current_combo = length_combo
        
        spell_set = SpellSet(length_combo, target_profile, spell_data)
        if spell_set.optimize_score(target_score, tolerance, progress=progress):
            valid_solutions.append(spell_set.get_words_as_strings())
            progress.solutions_found += 1
            progress.update(spell_set.get_total_score(), 
                          abs(spell_set.get_total_score() - target_score),
                          0, spell_set.get_words_as_strings(), force=True)
    
    return valid_solutions

def load_spell_data(csv_file):
    """
    Load spell data from CSV and process probabilities into scores.
    Expected format: Spell,Probability
    where Probability is in percentage format (e.g., "45.126%")
    """
    df = pd.read_csv(csv_file)
    df['Score'] = df['Probability'].str.rstrip('%').astype(float)
    return df

def get_spell_score(spell, spell_data):
    """Get the score for a spell from the data."""
    try:
        return spell_data.loc[spell_data['Spell'] == spell, 'Score'].iloc[0]
    except IndexError:
        raise ValueError(f"Spell '{spell}' not found in spell data")

def save_spell_sets(spell_sets, spell_data, output_file):
    """Save spell sets to a CSV file with additional rarity information."""
    rows = []
    set_number = 1
    
    for spell_set in spell_sets:
        # Add each spell in the set
        for spell in spell_set:
            score = get_spell_score(spell, spell_data)
            rows.append({
                'Set_Number': set_number,
                'Spell': spell,
                'Length': len(spell),
                'Score': score,
            })
        
        # Add sum row for the set
        total_length = sum(len(spell) for spell in spell_set)
        total_score = sum(get_spell_score(spell, spell_data) for spell in spell_set)
        rows.append({
            'Set_Number': set_number,
            'Spell': 'SUM',
            'Length': total_length,
            'Score': total_score,
        })
        
        set_number += 1
    
    # Create and save DataFrame
    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

def main():
    # Load the spell data
    spell_data = load_spell_data('tile_probabilities.csv')
    
    # Find equivalent spell sets
    print("Finding equivalent spell sets...")
    spell_sets = find_equivalent_spell_set(spell_data)
    
    # Save results
    if spell_sets:
        save_spell_sets(spell_sets, spell_data, 'equivalent_spell_sets.csv')
    else:
        print("No valid solutions found")

if __name__ == "__main__":
    main()