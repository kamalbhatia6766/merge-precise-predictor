"""
Shared pattern definitions for the prediction system.
Pure constants and helper functions - no side effects.
UPDATED: Corrected S40, 164950 family, and full k×k digit-pack universe
"""

import itertools
import math
from typing import Set, List, Dict, Tuple
from utils_2digit import to_2d_str

# S40 numbers (as zero-padded strings) - CORRECTED
S40_STRINGS = {
    '00', '06', '07', '09', '15', '16', '18', '19', '24', '25', '27', '28', 
    '33', '34', '36', '37', '42', '43', '45', '46', '51', '52', '54', '55', 
    '60', '61', '63', '64', '70', '72', '73', '79', '81', '82', '88', '89', 
    '90', '91', '97', '98'
}

# S40 numbers as integers for backward compatibility
S40 = [0, 6, 7, 9, 15, 16, 18, 19, 24, 25, 27, 28, 33, 34, 36, 37, 42, 43, 
       45, 46, 51, 52, 54, 55, 60, 61, 63, 64, 70, 72, 73, 79, 81, 82, 88, 
       89, 90, 91, 97, 98]

# 164950 family digits
PACK_164950_FAMILY = {'0', '1', '4', '5', '6', '9'}
SPECIAL_164950_DIGITS = {0, 1, 4, 5, 6, 9}  # For backward compatibility
SPECIAL_164950_ID = "pack6_014569"

# Generate full pack universe
DIGITS = list(range(10))
DIGIT_STRINGS = [str(d) for d in DIGITS]

def is_s40(number) -> bool:
    """Check if a number is in S40 set."""
    try:
        num_str = to_2d_str(number)
        return num_str in S40_STRINGS
    except Exception:
        return False

def is_164950_family(number) -> bool:
    """Check if a number belongs to 164950 family."""
    try:
        num_str = to_2d_str(number)
        d1, d2 = num_str[0], num_str[1]
        return (d1 in PACK_164950_FAMILY) and (d2 in PACK_164950_FAMILY)
    except Exception:
        return False

def get_kxk_pack_count(k: int) -> int:
    """Calculate number of k×k packs for given k."""
    if k < 2 or k > 6:
        return 0
    comb = math.comb(10, k)
    return comb * comb

def get_pack_universe_stats() -> Dict:
    """Get theoretical pack universe statistics."""
    k_packs = {}
    total_digit_packs = 0
    
    for k in [2, 3, 4, 5, 6]:
        count = get_kxk_pack_count(k)
        k_packs[k] = count
        total_digit_packs += count
    
    stats = {
        "k_packs": k_packs,
        "total_digit_packs": total_digit_packs,
        "special_packs": {
            "S40": 1,
            "PACK_164950": 1,
        },
        "total_entities": total_digit_packs + 1,  # +1 for S40
        "note": "PACK_164950 is one of the 6x6 packs but also tracked as special."
    }
    
    return stats

def run_pack_universe_sanity_check() -> Tuple[bool, str]:
    """Run comprehensive sanity check on pack universe."""
    try:
        stats = get_pack_universe_stats()
        
        # Check S40
        if len(S40) != 40:
            return False, f"S40 count incorrect: expected 40, got {len(S40)}"
        
        # Check 164950 family count
        count_164950 = sum(1 for n in range(100) if is_164950_family(n))
        expected_164950 = 6 * 6  # 6 digits × 6 digits
        if count_164950 != expected_164950:
            return False, f"164950 family count incorrect: expected {expected_164950}, got {count_164950}"
        
        # Check pack counts
        expected_total = 168129  # Pre-calculated total
        if stats["total_digit_packs"] != expected_total:
            return False, f"Total packs incorrect: expected {expected_total}, got {stats['total_digit_packs']}"
        
        # Build summary message
        message_lines = [
            "✅ Pack universe sanity: OK",
            f"   k-packs: k=2 → {stats['k_packs'][2]}, k=3 → {stats['k_packs'][3]}, k=4 → {stats['k_packs'][4]}, k=5 → {stats['k_packs'][5]}, k=6 → {stats['k_packs'][6]}",
            f"   Total digit packs: {stats['total_digit_packs']}",
            f"   Special packs: S40=1, PACK_164950=1 (inside 6x6)",
            f"   Total entities (digit-universe + S40): {stats['total_entities']}"
        ]
        
        return True, "\n".join(message_lines)
        
    except Exception as e:
        return False, f"❌ Pack universe check failed: {e}"

# Backward compatibility functions
def get_digit_pack_tags(n: int) -> List[str]:
    """Get digit pack tags for a number."""
    tags = []
    
    try:
        num = int(n)
        if 0 <= num <= 99:
            num_str = to_2d_str(num)
            
            # Add S40 tag
            if is_s40(num):
                tags.append("S40")
            
            # Add 164950 tag
            if is_164950_family(num):
                tags.append("PACK_164950")
            
            # Add coarse range packs for backward compatibility
            if num <= 19:
                tags.append("PACK_00_19")
            elif num <= 39:
                tags.append("PACK_20_39")
            elif num <= 59:
                tags.append("PACK_40_59")
            elif num <= 79:
                tags.append("PACK_60_79")
            else:
                tags.append("PACK_80_99")
            
            # Always add 2-digit tag
            tags.append("PACK_2DIGIT")
    
    except (ValueError, TypeError):
        pass
    
    return tags

def get_number_families(number) -> List[str]:
    """Get number families (alias for get_digit_pack_tags)."""
    return get_digit_pack_tags(number)

def print_pack_universe_summary():
    """Print comprehensive pack universe summary."""
    ok, msg = run_pack_universe_sanity_check()
    stats = get_pack_universe_stats()
    
    print("\n" + "="*60)
    print("🧮 PACK UNIVERSE SUMMARY & SANITY CHECK")
    print("="*60)
    print(msg)
    
    if ok:
        print("\n📊 DETAILED STATS:")
        for k, count in stats["k_packs"].items():
            print(f"   K={k}: {count} packs")
        print(f"   Total k×k packs: {stats['total_digit_packs']}")
        print(f"   Special packs: {stats['special_packs']}")
        print(f"   Total entities: {stats['total_entities']}")
        print("🎉 PACK UNIVERSE IS HEALTHY!")
    else:
        print("⚠️  PACK UNIVERSE HAS ISSUES!")
    
    print("="*60)

# Backward compatibility - keep existing pattern families
PACK_3_FAMILIES = ["123", "234", "345", "456", "567", "678", "789", "890"]
PACK_4_FAMILIES = ["1234", "2345", "3456", "4567", "5678", "6789", "7890"]
PACK_164950_NUMBERS = [n for n in range(100) if is_164950_family(n)]

if __name__ == "__main__":
    print_pack_universe_summary()
    
    # Test a few numbers
    print("\n🔍 TEST NUMBERS:")
    test_numbers = [0, 14, 15, 23, 55, 64, 79, 98]
    for num in test_numbers:
        tags = get_digit_pack_tags(num)
        s40 = is_s40(num)
        fam164950 = is_164950_family(num)
        print(f"  {num:02d}: S40={s40}, 164950={fam164950}, Tags={tags}")
