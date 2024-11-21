import csv
from collections import Counter

def parse_spell_data(csv_file):
    # Read CSV and group by set number
    spell_sets = {}
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            set_num = int(row['Set'])
            
            # Skip SUM rows
            if row['Spell'] == 'SUM':
                continue
                
            if set_num not in spell_sets:
                spell_sets[set_num] = {
                    'profile': row['Profile'],
                    'spells': []
                }
            
            spell_sets[set_num]['spells'].append({
                'spell': row['Spell'],
                'length': int(row['Length']),
                'score': float(row['Score'])
            })
    
    all_summaries = []
    
    # Generate summary for each set
    for set_num in sorted(spell_sets.keys()):
        set_data = spell_sets[set_num]
        spells = set_data['spells']
        
        # Calculate length distribution
        length_dist = Counter(spell['length'] for spell in spells)
        
        # Calculate letter distribution
        all_letters = ''.join(spell['spell'] for spell in spells)
        letter_dist = Counter(all_letters)
        
        # Calculate totals
        total_score = sum(spell['score'] for spell in spells)
        total_length = sum(spell['length'] for spell in spells)
        
        # Generate summary
        summary = f"""Spell Set #{set_num} - {set_data['profile']}
------------------------------
Spells:
"""
        for spell in spells:
            summary += f"  {spell['spell']:<8} - Length: {spell['length']}, Score: {spell['score']:.3f}\n"
        
        summary += f"""
Summary:
  Total Score: {total_score:.3f}
  Total Length: {total_length}

Length Distribution:
"""
        for length in sorted(length_dist.keys()):
            summary += f"  Length {length}: {length_dist[length]} words\n"
        
        summary += f"""
Letter Distribution:
"""
        for letter in sorted(letter_dist.keys()):
            summary += f"  {letter}: {letter_dist[letter]}\n"
        
        summary += "\n=================================================="
        
        all_summaries.append(summary)
    
    return "\nSpell Set Summary\n==================================================\n\n" + "\n\n".join(all_summaries)

def main():
    result = parse_spell_data('spell_sets.csv')
    
    # Write to output file
    with open('spell_set_summary.txt', 'w') as f:
        f.write(result)

if __name__ == "__main__":
    main()